# Large resident renders: live voice coordination diagnosis

- Document the live signal check and a two-request 5472x3072 reproduction in `docs/recipes/minimax-h3-image-metal.md`.
- The live voice request completed in 72.854 seconds while neither the shared speech lock nor its generation announced it. The existing render-side give-way cannot react to an absent signal; an HTTP 200 and early headers do not establish the voice-note deadline.
- A diagnostic client-owned shared hold reduced the same complete voice response to 7.473 seconds (first PCM 1.804 seconds). Both queued renders completed, with both latents and PNGs byte-identical to the uncoordinated run. This establishes the renderer's existing behavior, not a production fix.
- No scheduler or arithmetic changes: the live speech deployment must publish card 49's existing protocol on the same lock path. Full voice-menu acceptance remains blocked on owner deployment and a repeat of the original workload.
