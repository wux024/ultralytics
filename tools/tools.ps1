# Define an array containing all the sub-optical field sizes to test
$optical_field_sizes = @(16)
$dataset = "horse10"

# Loop through each sub-optical field size
foreach ($size in $optical_field_sizes) {

    # Build and run the command with the current sub-optical field size
    python tools/pose_stream_inference.py --dataset $dataset `
        --imgsz 256 `
        --models n `
        --model-type spipose `
        --optical-field-sizes 128 `
        --kpt-line `
        --sub-optical-field-sizes $size `
        --imgsz-hadamard 256 `
        --hadamard-seed 20241214 `
        --seed 20241216 `
}