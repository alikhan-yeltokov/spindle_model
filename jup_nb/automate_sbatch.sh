
#!/bin/bash
# Generate and submit 72 strings with different x, y, and z values
for x in {50..50..5}; do          # x: 5, 10, 15, 20, 25, 30
  for y in {40..40..20}; do      # y: 20, 40, 60, 80, 100, 120
    for z in {0..1}; do           # z: 0, 1
      # Create the string name
      name="spindle_9214_cell_FE_AL_1_0.3_SL_1.8_ts_0.05_opt_forces_3.6_4_2.5_MUD_${x}_MT_${y}_push_${z}"
      echo "Submitting: $name"
      # Submit the job with the generated name
      sbatch slurm_array_FE.sh "$name"
    done
  done
done



# for x in {5..30..5}; do          # x: 5, 10, 15, 20, 25, 30
#   for y in {20..120..20}; do      # y: 20, 40, 60, 80, 100, 120
#     for z in {0..1}; do           # z: 0, 1
#       # Create the string name
#       name="spindle_9214_FE_AL_1_0.3_SL_1.9_ts_0.05_opt_pull_3.6_PINS_${x}_MT_${y}_push_${z}"
#       echo "Submitting: $name"
#       # Submit the job with the generated name
#       sbatch slurm_array_FE.sh "$name"
#     done
#   done
# done