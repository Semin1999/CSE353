import numpy as np
import numpy.random as nr
import cv2

class MyCNN:
    # You may freely modify the constructor in any way you want.
    def __init__(self):
        # Initialize filters for three convolutional layers
        self.conv1_filters = nr.rand(3, 5, 5, 16) * 0.1  # (ni, fh, fw, no)
        self.conv2_filters = nr.rand(16, 5, 5, 32) * 0.1
        self.conv3_filters = nr.rand(32, 5, 5, 64) * 0.1

    # Implement convolution operation
    def conv(self, img, filters, stride=1, padding=0):
        bs, nc, h_in, w_in = img.shape
        ni, fh, fw, no = filters.shape
        assert nc == ni, "Number of input channels must match filter's input channels"

        # Add padding to the input image
        img_padded = np.pad(img, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        h_in_padded, w_in_padded = img_padded.shape[2], img_padded.shape[3]

        # Compute output dimensions
        h_out = int((h_in_padded - fh) / stride) + 1
        w_out = int((w_in_padded - fw) / stride) + 1

        # Initialize output tensor
        output = np.zeros((bs, no, h_out, w_out))

        # Perform convolution
        for b in range(bs):
            for o in range(no):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * stride
                        h_end = h_start + fh
                        w_start = j * stride
                        w_end = w_start + fw
                        img_patch = img_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, o, i, j] = np.sum(img_patch * filters[:, :, :, o])

        return output

    # Implement pooling operation
    def pool(self, img, size=2, stride=2, padding=0, ptype='max'):
        bs, nc, h_in, w_in = img.shape

        # Add padding if needed
        img_padded = np.pad(img, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        h_in_padded, w_in_padded = img_padded.shape[2], img_padded.shape[3]

        # Compute output dimensions
        h_out = int((h_in_padded - size) / stride) + 1
        w_out = int((w_in_padded - size) / stride) + 1

        # Initialize output tensor
        output = np.zeros((bs, nc, h_out, w_out))

        for b in range(bs):
            for c in range(nc):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * stride
                        h_end = h_start + size
                        w_start = j * stride
                        w_end = w_start + size
                        img_patch = img_padded[b, c, h_start:h_end, w_start:w_end]
                        if ptype == 'max':
                            output[b, c, i, j] = np.max(img_patch)
                        elif ptype == 'avg':
                            output[b, c, i, j] = np.mean(img_patch)
                        else:
                            raise ValueError("Invalid pooling type")

        return output

    # Implement ReLU activation function
    def relu(self, x):
        return np.maximum(0, x)

    # Implement forward pass of the network
    def forward(self, x):
        # Assuming x has shape (bs, 3, h, w)
        x = self.conv(x, self.conv1_filters, stride=1, padding=2)
        x = self.pool(x, size=2, stride=2, padding=0, ptype='max')
        x = self.relu(x)

        x = self.conv(x, self.conv2_filters, stride=1, padding=2)
        x = self.pool(x, size=2, stride=2, padding=0, ptype='max')
        x = self.relu(x)

        x = self.conv(x, self.conv3_filters, stride=1, padding=2)
        x = self.pool(x, size=2, stride=2, padding=0, ptype='max')
        x = self.relu(x)

        return x


# Below is just a test code for your reference.
# This is one way to use the conv() and pool() methods above
def main(imfile='semin.JPG', outname='output.png'):
    bs = 1     # Minibatch size set to 1 for a single image
    size = 400  # Input image size (w, h)
    filter_size = 5  # Choose your preferred size
    num_out_maps = 10  # Number of output feature maps
    num_in_maps = 3

    img = cv2.imread(imfile)   # Returns a numpy ndarray of shape (h, w, num_channel)
    img = cv2.resize(img, (size, size))
    img = np.transpose(img, (2, 0, 1))  # Permute the dimensions so the image shape is 3x400x400
    img = img[np.newaxis, :, :, :]  # Add batch dimension, shape becomes 1x3x400x400

    net = MyCNN()
    output = net.forward(img)  # Run the forward pass

    # Save 5 output feature maps
    for i in range(5):
        feature_map = output[0, i, :, :]  # Select the i-th feature map of the first image in batch
        # Normalize the feature map to [0,255]
        feature_map = feature_map - feature_map.min()
        if feature_map.max() != 0:
            feature_map = feature_map / feature_map.max()
        feature_map = (feature_map * 255).astype(np.uint8)
        cv2.imwrite(f"{outname}_{i}.png", feature_map)


if __name__ == '__main__':
    main()
