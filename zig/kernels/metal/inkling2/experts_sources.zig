//! Inkling 2 Mini's M3 packed-weight expert kernels; activations stay BF16 throughout.
pub const experts = @embedFile("experts.metal");
/// The same products on simdgroup matrices, for GPUs without tensor units: after the decode helpers above.
pub const experts_mma = experts ++ @embedFile("experts_mma.metal");
/// M5 tensor-unit products; the non-M5 simdgroup and scalar libraries stay independent.
pub const experts_units = @embedFile("../nax.h") ++ experts ++ @embedFile("units.metal") ++ @embedFile("experts_units.metal");
