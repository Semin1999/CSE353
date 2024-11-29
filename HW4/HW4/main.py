import numpy as np
import numpy.random as nr
import cv2 # You don't have to use this package, although I find it useful to have it around.
class MyCNN:
    # You may freely modify the constructor in any way you want.
    def init(self):
        ...
    # TODO: Implement conv() and pool(). Should be able to handle various values of stride and padding.
    # The size of 'img' is bs x nc x w x h (bs == batch size, nc == number of channels)
    # The size of 'filters' is ni x w x h x no, where 'ni' is the number of input channels and 'no' is the number of output channels.
    def conv(self, img, filters, stride=1, padding=0):
        ...
    # TODO: Implement pooling. The 'ptype' variable can be either 'max' or 'avg' (max pooling and average pooling).
    def pool(self, img, size=2, stride=2, padding=0, ptype='max'):
        ...
    # TODO: Implement a conv-pool-relu-conv-pool-relu-conv-pool-relu network using the above two methods.
    # The input 'x' should have dimensionality num_channels x 400 x 400.
    def forward(self, x):
        ...
# Below is just a test code for your reference.
# This is one way to use the conv() and pool() methods above
def main(imfile='test.jpg', outname='output'):
    bs = 16     # Minibatch size
    size = 400  # Input image size (w, h)
    filter_size = 5 # Choose your preferred size
    num_out_maps = 10 # Number of output feature maps.
    num_in_maps = 3
    img = cv2.imread(imfile)   # Returns a numpy ndarray of shape (w, h, num_channel)
    img = cv2.resize(img, (size, size))
    img = np.transpose(img, (2, 0, 1)) # Permute the dimensions so the image shape is 3x400x400
    filters = nr.rand(num_in_maps, filter_size, filter_size, num_out_maps)
    net = MyCNN()
    conv_out = net.conv(img, filters)
    # TODO: Make 'net.forward(img)' work
if name == 'main':
    main()