import numpy as np
import numpy.random as nr
import cv2  # You don't have to use this package, although I find it useful to have it around.

class MyCNN:
    # You may freely modify the constructor in any way you want.
    def init(self):
        ...

    # TODO: Implement conv() and pool(). Should be able to handle various values of stride and padding.
    # The size of 'img' is bs x nc x w x h (bs == batch size, nc == number of channels)
    # The size of 'filters' is ni x w x h x no, where 'ni' is the number of input channels and 'no' is the number of output channels.
    def conv(self, img, filters, stride=1, padding=0):
        # get size of img and filters
        bs, nc, h, w = img.shape
        nf, _, fh, fw = filters.shape

        # if padding > 0, get padding
        if padding > 0:
            padded_img = np.pad(img, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        else:
            padded_img = img

        # calculate output size
        out_h = (h - fh + 2 * padding) // stride + 1
        out_w = (w - fw + 2 * padding) // stride + 1

        # initialize output with zero
        output = np.zeros((bs, nf, out_h, out_w))

        # Convolution step
        for b in range(bs):  # each batch
            for k in range(nf):  # each filter/output channel
                for i in range(out_h):  # each height
                    for j in range(out_w):  # each width
                        # Find the input region covered by the filter
                        h_start = i * stride
                        h_end = h_start + fh
                        w_start = j * stride
                        w_end = w_start + fw
                        # Extract the region from the input
                        input_slice = padded_img[b, :, h_start:h_end, w_start:w_end]

                        # Apply the filter to the input region and sum the result
                        output[b, k, i, j] = np.sum(input_slice * filters[k, :, :, :])

        return output

    # TODO: Implement pooling. The 'ptype' variable can be either 'max' or 'avg' (max pooling and average pooling).
    def pool(self, img, size=2, stride=2, padding=0, ptype='max'):
        # get img shape
        bs, nc, h, w = img.shape

        # padding
        if padding > 0:
            padded_img = np.pad(img,((0, 0), (0, 0), (padding, padding), (padding, padding)))
        else:
            padded_img = img

        # calculate output size
        out_h = (h - size + 2 * padding) // stride + 1
        out_w = (w - size + 2 * padding) // stride + 1

        # initialize output with zero
        output = np.zeros((bs, nc, out_h, out_w))

        # max pooling
        if ptype == 'max':
            for b in range(bs):
                for c in range(nc):
                    for i in range(out_h):
                        for j in range(out_w):
                            h_start = i * stride
                            h_end = h_start + size
                            w_start = j * stride
                            w_end = w_start + size
                            # find max area
                            pool_region = padded_img[b, c, h_start:h_end, w_start:w_end]
                            output[b, c, i, j] = np.max(pool_region)

        # average pooling
        elif ptype == 'avg':
            for b in range(bs):
                for c in range(nc):
                    for i in range(out_h):
                        for j in range(out_w):
                            h_start = i * stride
                            h_end = h_start + size
                            w_start = j * stride
                            w_end = w_start + size
                            # calculate average
                            pool_region = padded_img[b, c, h_start:h_end, w_start:w_end]
                            output[b, c, i, j] = np.mean(pool_region)

        return output

    def relu(self, x):
        return np.maximum(0, x)

    # TODO: Implement a conv-pool-relu-conv-pool-relu-conv-pool-relu network using the above two methods.
    # The input 'x' should have dimensionality num_channels x 400 x 400.
    def forward(self, x):
        # Add batch dimension
        x = np.expand_dims(x, 0)  # Now x has shape (1, num_channels, height, width)

        # 1st conv-pool-relu
        num_filters1 = 2
        num_channels = x.shape[1]  # Should be 3 for RGB image

        # Define edge detection filters for all channels
        h_edge_filter = np.array([[-1, -1, -1],
                                  [0, 0, 0],
                                  [1, 1, 1]])
        v_edge_filter = np.array([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]])

        filters1 = np.zeros((num_filters1, num_channels, 3, 3))
        for c in range(num_channels):
            filters1[0, c, :, :] = h_edge_filter
            filters1[1, c, :, :] = v_edge_filter

        conv1 = self.conv(x, filters1)
        pool1 = self.pool(conv1)
        relu1 = self.relu(pool1)

        # 2nd conv-pool-relu
        num_filters2 = 32
        filters2 = np.random.randn(num_filters2, relu1.shape[1], 3, 3) * 0.01
        conv2 = self.conv(relu1, filters2)
        pool2 = self.pool(conv2)
        relu2 = self.relu(pool2)

        # 3rd conv-pool-relu
        num_filters3 = 64
        filters3 = np.random.randn(num_filters3, relu2.shape[1], 3, 3) * 0.01
        conv3 = self.conv(relu2, filters3)
        pool3 = self.pool(conv3)
        relu3 = self.relu(pool3)

        return relu3

# Below is just a test code for your reference.
# This is one way to use the conv() and pool() methods above
def main(imfile='test.jpg', outname='output'):
    bs = 16     # Minibatch size adjusted to 1
    size = 400  # Input image size (w, h)

    img = cv2.imread(imfile)   # Returns a numpy ndarray of shape (w, h, num_channel)
    img = cv2.resize(img, (size, size))
    img = np.transpose(img, (2, 0, 1)) # Permute the dimensions so the image shape is 3x400x400
    net = MyCNN()
    output = net.forward(img)

    # TODO: Make 'net.forward(img)' work
    # extract 5 feature maps and save them
    feature_maps = output[0, :5, :, :]  # First 5 feature maps from the first image in the batch
    for i in range(5):
        fmap = feature_maps[i]
        # Normalize feature map to [0, 255] for saving as image
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min()) * 255
        fmap = fmap.astype(np.uint8)
        # Save the feature map
        cv2.imwrite(f"{outname}_feature_map_{i}.png", fmap)

if __name__ == '__main__':
    main()