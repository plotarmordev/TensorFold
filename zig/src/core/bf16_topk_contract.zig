//! Re-export the host contract without importing its source into a second Zig module.
const contract = @import("core").bf16_topk;
pub const max_k = contract.max_k;
pub const max_rows = contract.max_rows;
pub const Row = contract.Row;
pub const Params = contract.Params;
pub const check = contract.check;
