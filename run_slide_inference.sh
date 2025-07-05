#!/bin/bash


start_time=$(date +%s)
echo "Starting script at: $(date)"


# Check if GPU argument is provided
if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU must be an integer."
    exit 1
fi


GPU=$1

python generate_slide_embedding.py --gpu_id "$GPU" --total_gpus 2



end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Script finished at: $(date)"
echo "Total duration: $((duration / 60)) minutes and $((duration % 60)) seconds"