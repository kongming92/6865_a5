import sys
sys.path.append('utils')
import imageIO as io
import numpy as np
import a5

def test_computeWeight():
  im=io.imread('data/design-2.png')
  out=a5.computeWeight(im)
  io.imwrite(out, 'design-2_mask.png')

def test_computeFactor():
  im2=io.imread('data/design-2.png')
  im3=io.imread('data/design-3.png')
  w2=a5.computeWeight(im2)
  w3=a5.computeWeight(im3)
  out=a5.computeFactor(im2, w2, im3, w3)
  if abs(out-50.8426955376)<1 :
    print 'Correct'

def test_makeHDR(file):
  import glob
  inputs=sorted(glob.glob('data/' + file + '-*.png'))
  im_list = []
  for inp in inputs:
    im_list.append(io.imread(inp))

  hdr=a5.makeHDR(im_list)
  np.save('npy/'+file+'hdr', hdr)

  hdr_scale=hdr/max(hdr.flatten())
  io.imwrite(hdr_scale, file+'_hdr_linear_scale.png')

def test_toneMap(file):
  hdr=np.load('npy/'+file+'hdr.npy')
  out1 = a5.toneMap(hdr, 100, 1.0, False)
  io.imwrite(out1, file+'tone_map_gauss.png')
  out2 = a5.toneMap(hdr, 100, 3.0, True)
  io.imwrite(out2, file+'tone_map_bila.png')


# Uncomment the below to test your code

# test_computeWeight()
# test_computeFactor()
images = ['ante1', 'ante2', 'ante3', 'nyc', 'sea']
# images = ['design', 'horse', 'stairs', 'vine']
for f in images:
  print 'Testing', f
  test_makeHDR(f)
  test_toneMap(f)


