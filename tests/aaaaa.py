from src.move_encoding import uci_to_flat_index, flat_index_to_uci



print(uci_to_flat_index("e2e4"))      # Debe ser un número entre 0 y 4671
print(uci_to_flat_index("e7e8q"))     # Debe ser válido
print(flat_index_to_uci(0))           # No debe ser vacío