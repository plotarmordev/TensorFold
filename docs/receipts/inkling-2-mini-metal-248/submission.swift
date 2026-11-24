// One Metal submission's host cost: commit an empty command buffer with a blit encoder, wait, time it. 200 runs,
// the median printed; a round's saving is the submissions it no longer makes. Run: `swift submission.swift`.
import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let n = 200
var samples = [Double]()
samples.reserveCapacity(n)
for _ in 0..<n {
    let cb = queue.makeCommandBuffer()!
    let blit = cb.makeBlitCommandEncoder()!
    blit.endEncoding()
    let t0 = Double(DispatchTime.now().uptimeNanoseconds)
    cb.commit()
    cb.waitUntilCompleted()
    samples.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000.0)
}
samples.sort()
print(String(format: "submission+wait: median %.1f us, p95 %.1f us, min %.1f us", samples[n / 2], samples[n * 19 / 20], samples[0]))
