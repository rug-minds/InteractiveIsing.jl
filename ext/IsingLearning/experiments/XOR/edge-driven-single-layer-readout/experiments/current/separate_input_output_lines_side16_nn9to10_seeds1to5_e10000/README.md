# Long Edge Readout Sweep, NN 9-10

This folder contains the long edge-driven XOR runs for NN `9` and `10`, seeds
`1:5`, using the separate-line architecture:

```text
16x1 input line -> 16x16 dynamic field -> 16x1 output line
```

The input and output lines are separate layers coupled to the left and right
edges of the dynamic field. They are not frozen columns inside the `16x16`
field.

## Readout

The current readout is `two_class`, not a full-line majority vote. The final
`16x1` output line is split into two replicated class regions:

- sites `1:8`: XOR false class
- sites `9:16`: XOR true class

The scalar score is:

```text
score = mean(output[9:16]) - mean(output[1:8])
```

The prediction is XOR true when `score > 0`. The logged `mse` compares this
score to the target `-1` or `+1`, and `min_margin` is the weakest signed score
over the four XOR cases.
