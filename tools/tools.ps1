# Define an array containing all the sub-optical field sizes to test
$optical_field_sizes = @(128)
$dataset = "twoperson"

# Loop through each sub-optical field size
foreach ($size in $optical_field_sizes) {

    # Build and run the command with the current sub-optical field size
    python tools/pose_stream_inference.py --dataset $dataset `
        --imgsz 640 `
        --models n `
        --model-type spipose `
        --optical-field-sizes $size `
        --kpt-line `
        --sub-optical-field-size 16 `
        --imgsz-hadamard 256 `
        --hadamard-seed 20241128 `
        --seed 20241205 `
}