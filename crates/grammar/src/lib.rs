//! Constrained decoding with a grammar. A regular expression (or a JSON
//! schema, turned into one by `json_schema`) is compiled into a DFA over
//! bytes. At each step the allowed tokens are those whose bytes keep the DFA
//! alive, worked out once per DFA state and cached; the end-of-sequence
//! tokens are allowed once the output so far is a full match.

pub mod json_schema;

use anyhow::{Context, Result};
use regex_automata::dfa::{Automaton, dense};
use regex_automata::util::primitives::StateID;
use regex_automata::util::start;
use regex_automata::{Anchored, MatchKind};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// What the grammar needs to know about the vocabulary: every token's bytes
/// (`None` for special tokens, which text never contains), and which tokens
/// end a sequence.
pub struct Vocab {
    pub tokens: Vec<Option<Vec<u8>>>,
    pub eos: Vec<u32>,
}

impl Vocab {
    /// Token strings of a byte-level BPE tokenizer (GPT-2 style: Qwen, Llama
    /// 3, SmolLM2) back to the bytes they stand for. A string with a
    /// character outside that alphabet is not a byte-level token: `None`.
    pub fn from_byte_level(strings: Vec<Option<String>>, eos: Vec<u32>) -> Self {
        let mut unicode_to_byte = HashMap::new();
        let mut n = 0u32;
        for b in 0..=255u8 {
            let printable = (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
            let c = if printable {
                b as u32
            } else {
                n += 1;
                255 + n
            };
            unicode_to_byte.insert(char::from_u32(c).unwrap(), b);
        }
        let tokens = strings
            .into_iter()
            .map(|s| s?.chars().map(|c| unicode_to_byte.get(&c).copied()).collect())
            .collect();
        Self { tokens, eos }
    }
}

/// A compiled grammar, shared by every matcher of it.
pub struct Grammar {
    dfa: dense::DFA<Vec<u32>>,
    vocab: Arc<Vocab>,
    /// Allowed tokens per DFA state, filled in as states are reached.
    allowed: Mutex<HashMap<StateID, Arc<Vec<u32>>>>,
}

impl Grammar {
    /// Output must match `pattern` in full.
    pub fn regex(pattern: &str, vocab: Arc<Vocab>) -> Result<Arc<Self>> {
        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .start_kind(regex_automata::dfa::StartKind::Anchored)
                    .match_kind(MatchKind::All),
            )
            // Anchored at the end too: only the end of the text can match, so a
            // byte that breaks the pattern leads to the dead state right away.
            .build(&format!(r"(?:{pattern})\z"))
            .with_context(|| format!("bad pattern {pattern:?}"))?;
        Ok(Arc::new(Self {
            dfa,
            vocab,
            allowed: Mutex::default(),
        }))
    }

    /// Output must be JSON valid against `schema` (see `json_schema`).
    pub fn json_schema(schema: &str, vocab: Arc<Vocab>) -> Result<Arc<Self>> {
        let schema: serde_json::Value = serde_json::from_str(schema).context("schema is not JSON")?;
        Self::regex(&json_schema::to_regex(&schema)?, vocab)
    }

    pub fn start(self: &Arc<Self>) -> Result<Matcher> {
        let state = self.dfa.start_state(&start::Config::new().anchored(Anchored::Yes))?;
        Ok(Matcher {
            grammar: self.clone(),
            state,
        })
    }

    fn step(&self, mut state: StateID, bytes: &[u8]) -> Option<StateID> {
        for &b in bytes {
            state = self.dfa.next_state(state, b);
            if self.dfa.is_dead_state(state) {
                return None;
            }
        }
        Some(state)
    }

    fn complete(&self, state: StateID) -> bool {
        self.dfa.is_match_state(self.dfa.next_eoi_state(state))
    }

    fn allowed(&self, state: StateID) -> Arc<Vec<u32>> {
        if let Some(a) = self.allowed.lock().unwrap().get(&state) {
            return a.clone();
        }
        let mut allowed: Vec<u32> = (0..self.vocab.tokens.len() as u32)
            .filter(|&t| {
                let bytes = self.vocab.tokens[t as usize].as_deref();
                bytes.is_some_and(|b| !b.is_empty() && self.step(state, b).is_some())
            })
            .collect();
        if self.complete(state) {
            allowed.extend(&self.vocab.eos);
        }
        let allowed = Arc::new(allowed);
        self.allowed.lock().unwrap().insert(state, allowed.clone());
        allowed
    }
}

/// Where one output is in a grammar.
pub struct Matcher {
    grammar: Arc<Grammar>,
    state: StateID,
}

impl Matcher {
    /// The tokens that may come next.
    pub fn allowed(&self) -> Arc<Vec<u32>> {
        self.grammar.allowed(self.state)
    }

    /// Take `token` as the next one. Fails if the grammar does not allow it.
    pub fn accept(&mut self, token: u32) -> Result<(), String> {
        if self.grammar.vocab.eos.contains(&token) {
            return if self.complete() {
                Ok(())
            } else {
                Err("the output is not complete yet".into())
            };
        }
        let bytes = self.grammar.vocab.tokens.get(token as usize).and_then(|t| t.as_deref());
        let next = bytes.and_then(|b| self.grammar.step(self.state, b));
        self.state = next.ok_or_else(|| format!("token {token} does not fit the grammar"))?;
        Ok(())
    }

    /// Whether the output so far is a full match.
    pub fn complete(&self) -> bool {
        self.grammar.complete(self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Arc<Vocab> {
        let words = [
            "{",
            "}",
            "\"",
            "name",
            "age",
            ":",
            ",",
            "1",
            "2",
            "3",
            "0",
            "a",
            "b",
            "\"name\":",
            " ",
            "x",
        ];
        Arc::new(Vocab {
            tokens: words.iter().map(|w| Some(w.as_bytes().to_vec())).collect(),
            eos: vec![99],
        })
    }

    fn text(v: &Vocab, allowed: &[u32]) -> Vec<String> {
        allowed
            .iter()
            .map(|&t| {
                v.tokens
                    .get(t as usize)
                    .and_then(|b| b.clone())
                    .map_or("<eos>".into(), |b| String::from_utf8(b).unwrap())
            })
            .collect()
    }

    #[test]
    fn digits_then_end() {
        let v = vocab();
        let mut m = Grammar::regex("[0-9]+", v.clone()).unwrap().start().unwrap();
        assert_eq!(text(&v, &m.allowed()), ["1", "2", "3", "0"]);
        m.accept(7).unwrap();
        assert_eq!(text(&v, &m.allowed()), ["1", "2", "3", "0", "<eos>"]);
        assert!(m.accept(11).is_err());
    }

    #[test]
    fn json_object() {
        let v = vocab();
        let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}"#;
        let mut m = Grammar::json_schema(schema, v.clone()).unwrap().start().unwrap();
        assert_eq!(text(&v, &m.allowed()), ["{"]);
        for t in [0, 13, 2, 11, 2, 6, 2, 4, 2, 5, 8, 10, 1] {
            // {"name":"a","age":20}
            assert!(!m.complete());
            m.accept(t).unwrap();
        }
        assert!(m.complete());
    }

    #[test]
    fn json_allows_spacing() {
        let words = ["{", "}", "\"a\"", ":", " ", "1", "\n"];
        let v = Arc::new(Vocab {
            tokens: words.iter().map(|w| Some(w.as_bytes().to_vec())).collect(),
            eos: vec![],
        });
        let schema = r#"{"type":"object","properties":{"a":{"type":"integer"}}}"#;
        let mut m = Grammar::json_schema(schema, v).unwrap().start().unwrap();
        // {\n "a": 1\n}
        for t in [0, 6, 4, 2, 3, 4, 5, 6, 1] {
            m.accept(t).unwrap();
        }
        assert!(m.complete());
    }

    #[test]
    fn byte_level_alphabet() {
        // "Ġ" is how byte-level BPE writes a space.
        let v = Vocab::from_byte_level(vec![Some("Ġhello".into()), Some("Ã©".into()), None], vec![]);
        assert_eq!(v.tokens[0].as_deref(), Some(" hello".as_bytes()));
        assert_eq!(v.tokens[1].as_deref(), Some("é".as_bytes()));
        assert_eq!(v.tokens[2], None);
    }
}
