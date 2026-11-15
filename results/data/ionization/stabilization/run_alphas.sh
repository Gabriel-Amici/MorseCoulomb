for alpha in  0.6 0.7 0.8 0.9 1.1 1.2; do
    echo "Running alpha = $alpha"
    python ../../../../scripts/ionization/classical/classical_ionization_stabilization.py --alpha="$alpha" --workers=8
 done
