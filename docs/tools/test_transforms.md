# `tools/test_transforms.py`

## Purpose and place

Runs small synthetic tests for `tools/transforms.py`. It is a validation script, not a dataset-processing stage.

## Inputs and outputs

It creates an in-memory two-sample egomotion table and prints unittest results. It reads no dataset files and writes no outputs.

## Run

```powershell
python tools/test_transforms.py
```

## Walkthrough and math

`_df` builds timestamps 0 and 10 with translation changing from zero to `(10,2,4)` and a Z rotation changing to 90 degrees. Tests verify that a transform followed by its inverse is identity, that a known point rotates/translates as expected, and that the midpoint is `(5,1,2)` with a 45-degree rotation. Relative-pose composition is checked too.

## Constants and conventions

The test’s values are deliberately synthetic: timestamps 0/10, a 90-degree endpoint, and a half-way query. Changing them changes only the test scenario, not production configuration.

## Checks and failure modes

The test is useful for algebra and interpolation regressions, but it does not validate real calibration, ZIP parsing, or dataset conventions. A failed assertion identifies a transform helper regression.

## Explain it back

1. **Does it need downloaded data?** No.
2. **What endpoint rotation is tested?** 90 degrees about Z.
3. **What interpolation is tested?** Translation midpoint and rotational slerp.
4. **What is the identity check?** `T @ inv(T)`.
5. **What is not tested?** Real sensor rows and real timestamps.
