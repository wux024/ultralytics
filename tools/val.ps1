# Define an array containing all the sub-optical field sizes to test
$subOpticalFieldSizes = @('None')
$dataset = "twoperson"

# Loop through each sub-optical field size
foreach ($size in $subOpticalFieldSizes) {
    # Determine if --inverse should be included
    # $inverseFlag = if ($size -eq 'None' -or $size -eq 128) { "" } else { "--inverse" }

    # Build and run the command with the current sub-optical field size
    python tools/pose_val.py --dataset $dataset `
        --imgsz 640 `
        --save_json `
        --models n `
        --model-type spipose `
        --optical-field-sizes 128 `
        --imgsz-hadamard 256 `
        --seed 20241026

    # Optional: Output a message after each run to track progress
    Write-Output "Finished processing with sub-optical-field-sizes: $size and inverse: $inverseFlag"
}