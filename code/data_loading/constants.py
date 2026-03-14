"""
Shared physical and chemical constants for proton affinity calculations.
"""

# Unit conversions
HARTREE_TO_KCALMOL = 627.509474  # 1 Hartree in kcal/mol
KJMOL_TO_KCALMOL = 1.0 / 4.184  # 1 kJ/mol in kcal/mol
KCALMOL_TO_KJMOL = 4.184         # 1 kcal/mol in kJ/mol

# Proton thermochemistry
H_PROTON_HA = 0.00235827        # H(H+) = 5/2 RT at 298.15 K in Hartree (1.48 kcal/mol)
HOF_PROTON_KCALMOL = 365.7      # Heat of formation of H+ in kcal/mol (PM7 gas-phase standard)

# ZPE scaling
ZPE_SCALE_FACTOR = 0.9850       # Scott & Radom 1996, for B3LYP
