with open("data/modelnet40_normal_resampled/modelnet40_test_orig.txt") as f:
  lines = f.readlines()
lines = lines[::13]
with open("data/modelnet40_normal_resampled/modelnet40_test.txt", 'w') as f2:
  for line in lines:
    f2.write(line)
