from src.move_encoding import uci_to_flat_index   

print(uci_to_flat_index("e2e4"))     # Debe ser un número entre 0 y 4671
print(uci_to_flat_index("g1f3"))     # Debe ser válido
print(uci_to_flat_index("e7e8q"))    # Coronación