// blit.swift: the receipt's bandwidth ceiling input. A device-to-device blit of 4 GiB, best of 12 reps,
// GB/s counted as bytes read plus bytes written. Runs on any Mac: swift blit.swift.
import Metal
import Foundation

let reps = 12, mib = 4096
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let n = mib * 1024 * 1024
let src = dev.makeBuffer(length: n, options: .storageModePrivate)!
let dst = dev.makeBuffer(length: n, options: .storageModePrivate)!
var best = 0.0
for _ in 0..<reps {
    let cb = q.makeCommandBuffer()!
    let blit = cb.makeBlitCommandEncoder()!
    blit.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0, size: n)
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let s = cb.gpuEndTime - cb.gpuStartTime // GPU seconds, submission left out
    best = max(best, 2.0 * Double(n) / s)
}
print(String(format: "%.0f GB/s", best / 1e9), "best of", reps, "over a", mib, "MiB blit")
