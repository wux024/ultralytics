# Define an array containing all the sub-optical field sizes to test
$subOpticalFieldSizes = @(16)
$dataset = "twoperson"

# Loop through each sub-optical field size
foreach ($size in $subOpticalFieldSizes) {

    # Build and run the command with the current sub-optical field size
    python tools/pose_stream_inference.py --dataset $dataset `
        --imgsz 640 `
        --models n `
        --model-type spipose `
        --optical-field-sizes 128 `
        --sub-optical-field-sizes $size `
        --imgsz-hadamard 256 `
        --kpt-line `
        --seed 20241027

    # Optional: Output a message after each run to track progress
    Write-Output "Finished processing with sub-optical-field-sizes: $size and inverse: $inverseFlag"
}