import scipy.io

# Caminho para o ficheiro
mat_path = 'part_B/test_data/ground-truth/GT_IMG_1.mat'

# Lê o ficheiro .mat
mat = scipy.io.loadmat(mat_path)
print(mat)
# Extrai os pontos anotados (coordenadas das pessoas)
points = mat["image_info"][0][0][0][0][0]
print(f"Número real de pessoas: {len(points)}")
