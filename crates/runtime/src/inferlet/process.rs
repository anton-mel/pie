//! Processes: running programs, each with an id. A process outlives the
//! client that started it. What it sends while nobody listens is kept, any
//! client can attach to it later, send it messages or kill it, and how it
//! ends (returned, failed or killed) is recorded in one place, `emit`.

use super::{Host, Session};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use wasmtime::component::Component;

pub type ProcessId = u64;

/// What a process says: its messages, then how it ended.
#[derive(Clone, Debug)]
pub enum Event {
    Message(String),
    Exited(Result<String, String>),
}

pub struct ProcessInfo {
    pub id: ProcessId,
    pub program: String,
    pub running: bool,
}

pub struct Processes {
    host: Arc<Host>,
    next: AtomicU64,
    table: Mutex<HashMap<ProcessId, Arc<Process>>>,
}

pub struct Process {
    pub id: ProcessId,
    pub program: String,
    /// Where messages for it go; none once the input is closed.
    input: Mutex<Option<mpsc::UnboundedSender<String>>>,
    output: Mutex<Output>,
    task: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Default)]
struct Output {
    /// Events nobody has received yet.
    backlog: Vec<Event>,
    /// The attached client, if any, and which attachment it is.
    listener: Option<(u64, mpsc::UnboundedSender<Event>)>,
    attachments: u64,
    ended: bool,
}

impl Processes {
    /// UPDATED
    /// Ids count up from `first`: several workers give their processes
    /// ids that do not collide, and an id says which worker holds it.
    pub fn new(host: Arc<Host>, first: u64) -> Self {
        Self {
            host,
            next: AtomicU64::new(first),
            table: Mutex::default(),
        }
    }

    /// Start `component` as a new process.
    pub fn spawn(&self, component: Component, program: &str, args: Vec<String>) -> Arc<Process> {
        let (out, mut outbox) = mpsc::unbounded_channel();
        let (input, inbox) = mpsc::unbounded_channel();
        let session = Session {
            out,
            inbox: Arc::new(tokio::sync::Mutex::new(inbox)),
        };
        let process = Arc::new(Process {
            id: self.next.fetch_add(1, Ordering::Relaxed),
            program: program.to_string(),
            input: Mutex::new(Some(input)),
            output: Mutex::default(),
            task: Mutex::new(None),
        });

        let (host, p) = (self.host.clone(), process.clone());
        let task = tokio::spawn(async move {
            let run = host.run(&component, args, session);
            tokio::pin!(run);
            let result = loop {
                tokio::select! {
                    Some(message) = outbox.recv() => p.emit(Event::Message(message)),
                    result = &mut run => break result,
                }
            };
            while let Ok(message) = outbox.try_recv() {
                p.emit(Event::Message(message));
            }
            p.emit(Event::Exited(result.unwrap_or_else(|e| Err(e.to_string()))));
        });
        *process.task.lock().unwrap() = Some(task);
        self.table.lock().unwrap().insert(process.id, process.clone());
        process
    }

    pub fn get(&self, id: ProcessId) -> Option<Arc<Process>> {
        self.table.lock().unwrap().get(&id).cloned()
    }

    pub fn list(&self) -> Vec<ProcessInfo> {
        let mut list: Vec<ProcessInfo> = self
            .table
            .lock()
            .unwrap()
            .values()
            .map(|p| ProcessInfo {
                id: p.id,
                program: p.program.clone(),
                running: !p.output.lock().unwrap().ended,
            })
            .collect();
        list.sort_by_key(|p| p.id);
        list
    }

    /// Stop a process and forget it. Dropping its task drops its instance,
    /// and with it every page it held.
    pub fn kill(&self, id: ProcessId) -> bool {
        let Some(p) = self.table.lock().unwrap().remove(&id) else {
            return false;
        };
        if let Some(task) = p.task.lock().unwrap().take() {
            task.abort();
        }
        p.emit(Event::Exited(Err("killed".into())));
        true
    }

    /// Forget a process that has ended and whose end was delivered.
    pub fn reap(&self, id: ProcessId) {
        let mut table = self.table.lock().unwrap();
        if table.get(&id).is_some_and(|p| p.output.lock().unwrap().ended) {
            table.remove(&id);
        }
    }
}

impl Process {
    /// Hand an event to the attached client, or keep it for the next one.
    fn emit(&self, event: Event) {
        let mut output = self.output.lock().unwrap();
        if output.ended {
            return;
        }
        output.ended = matches!(event, Event::Exited(_));
        if let Some((_, listener)) = &output.listener {
            if listener.send(event.clone()).is_ok() {
                return;
            }
            output.listener = None;
        }
        output.backlog.push(event);
    }

    /// Listen to it: first everything it said while nobody listened, then
    /// what it says next. A new listener replaces the old one. Returns which
    /// attachment this is, for `detach`.
    pub fn attach(&self) -> (u64, mpsc::UnboundedReceiver<Event>) {
        let mut output = self.output.lock().unwrap();
        let (tx, rx) = mpsc::unbounded_channel();
        for event in output.backlog.drain(..) {
            let _ = tx.send(event);
        }
        output.attachments += 1;
        let attachment = output.attachments;
        output.listener = Some((attachment, tx));
        (attachment, rx)
    }

    /// Stop listening, if `attachment` is still the listener. Events it
    /// received but did not deliver, `undelivered`, are kept for the next.
    pub fn detach(&self, attachment: u64, undelivered: Vec<Event>) {
        let mut output = self.output.lock().unwrap();
        if output.listener.as_ref().is_some_and(|(a, _)| *a == attachment) {
            output.listener = None;
            let later = std::mem::take(&mut output.backlog);
            output.backlog = undelivered;
            output.backlog.extend(later);
        }
    }

    /// Give it a message (`session.receive`). False once its input is closed.
    pub fn send(&self, message: String) -> bool {
        self.input
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|i| i.send(message).is_ok())
    }

    /// No more messages: its next `session.receive` returns none.
    pub fn close_input(&self) {
        *self.input.lock().unwrap() = None;
    }
}
