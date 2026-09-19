//! `grammar`: output that must follow a grammar (`crates/grammar`).

use crate::inferlet::State;
use crate::inferlet::pie::inferlet::grammar;
use ::grammar::{Grammar, Matcher};
use wasmtime::component::Resource;

impl grammar::Host for State {}

impl grammar::HostMatcher for State {
    async fn from_regex(&mut self, pattern: String) -> Result<Resource<Matcher>, String> {
        let matcher = Grammar::regex(&pattern, self.engine.vocab()).and_then(|g| g.start());
        self.push_matcher(matcher)
    }

    async fn from_json_schema(&mut self, schema: String) -> Result<Resource<Matcher>, String> {
        let matcher = Grammar::json_schema(&schema, self.engine.vocab()).and_then(|g| g.start());
        self.push_matcher(matcher)
    }

    async fn allowed(&mut self, m: Resource<Matcher>) -> Vec<u32> {
        self.table.get(&m).map(|m| m.allowed().to_vec()).unwrap_or_default()
    }

    async fn accept(&mut self, m: Resource<Matcher>, token: u32) -> Result<(), String> {
        self.table.get_mut(&m).map_err(|e| e.to_string())?.accept(token)
    }

    async fn complete(&mut self, m: Resource<Matcher>) -> bool {
        self.table.get(&m).is_ok_and(|m| m.complete())
    }

    async fn drop(&mut self, m: Resource<Matcher>) -> wasmtime::Result<()> {
        self.table.delete(m)?;
        Ok(())
    }
}

impl State {
    fn push_matcher(&mut self, matcher: anyhow::Result<Matcher>) -> Result<Resource<Matcher>, String> {
        let matcher = matcher.map_err(|e| format!("{e:#}"))?;
        self.table.push(matcher).map_err(|e| e.to_string())
    }
}
