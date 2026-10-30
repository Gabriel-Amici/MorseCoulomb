for alpha in 0 0.01 0.1 0.5 1 1.1 1.25 1.5; do
    echo "Running alpha = $alpha"
    python ../../../scripts/ionization/classical/classical_ionization_vs_frequency.py --alpha="$alpha"
done
