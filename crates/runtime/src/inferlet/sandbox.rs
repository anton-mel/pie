//! What an inferlet may reach besides the model. By default nothing: it
//! gets stdio only. A policy can grant one directory, seen by the inferlet
//! as `/data`, and TCP connections to a list of addresses.

use anyhow::Result;
use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use wasmtime_wasi::sockets::SocketAddrUse;
use wasmtime_wasi::{FsPerms, WasiCtx};

#[derive(Clone, Default)]
pub struct Policy {
    /// A host directory the inferlet sees as `/data`.
    pub dir: Option<PathBuf>,
    /// Whether it may change files there, not only read them.
    pub writable: bool,
    /// The only addresses it may open TCP connections to.
    pub connect: Vec<SocketAddr>,
}

impl Policy {
    /// A WASI context that grants this policy and nothing more.
    pub fn wasi(&self) -> Result<WasiCtx> {
        let mut wasi = WasiCtx::builder();
        wasi.inherit_stdio();
        if let Some(dir) = &self.dir {
            let perms = if self.writable {
                FsPerms::ReadWrite
            } else {
                FsPerms::ReadOnly
            };
            wasi.preopened_dir(dir, "/data", perms)?;
        }
        if !self.connect.is_empty() {
            let allowed: Arc<HashSet<SocketAddr>> = Arc::new(self.connect.iter().copied().collect());
            wasi.allow_tcp(true).socket_addr_check(move |addr, used| {
                let ok = match used {
                    SocketAddrUse::TcpConnect => allowed.contains(&addr),
                    // Connecting first binds to any local address.
                    SocketAddrUse::TcpBind => addr.ip().is_unspecified(),
                    _ => false,
                };
                Box::pin(async move { ok })
            });
        }
        Ok(wasi.build())
    }
}
