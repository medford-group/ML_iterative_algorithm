from coulombmatrix import CM_descriptors
CM = CM_descriptors()
data = 'molecules_name.csv'
CM_fpt, columns = CM.CoulombMatrix(data)
sinCM_fpt, columns = CM.SinCoulombMatrix(data)

