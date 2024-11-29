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
        # 1. 입력 이미지의 크기 가져오기
        bs, nc, h, w = img.shape
        ni, fh, fw, no = filters.shape

        # 2. 패딩 적용
        if padding > 0:
            # numpy.pad를 사용하여 이미지에 패딩 추가
            # ((위,아래), (좌,우)) 각각에 padding 크기만큼 추가
            padded_img = np.pad(img, ((0, 0), (0, 0), (padding, padding), (padding, padding)),mode='constant')
        else:
            padded_img = img

        # 3. 출력 크기 계산
        out_h = (h - fh + 2 * padding) // stride + 1
        out_w = (w - fw + 2 * padding) // stride + 1

        output = np.zeros((bs, no, out_h, out_w))

        # 5. Convolution 연산 수행
        for b in range(bs):  # 각 배치에 대해
            for i in range(out_h):  # 출력 높이
                for j in range(out_w):  # 출력 너비
                    # 현재 위치에서의 입력 영역 추출
                    h_start = i * stride
                    h_end = h_start + fh
                    w_start = j * stride
                    w_end = w_start + fw

                    # 입력 영역 추출
                    input_slice = padded_img[b, :, h_start:h_end, w_start:w_end]

                    # 각 출력 채널에 대해 합성곱 연산
                    for k in range(no):  # 출력 채널
                        # sum over input channels
                        output[b, k, i, j] = np.sum(input_slice * filters[:, :, :, k])

        return output

    # TODO: Implement pooling. The 'ptype' variable can be either 'max' or 'avg' (max pooling and average pooling).
    def pool(self, img, size=2, stride=2, padding=0, ptype='max'):
        # 1. 입력 이미지의 크기 가져오기
        bs, nc, h, w = img.shape

        # 2. 패딩 적용
        if padding > 0:
            padded_img = np.pad(img,
                                ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                                mode='constant')
        else:
            padded_img = img

        # 3. 출력 크기 계산
        out_h = (h - size + 2 * padding) // stride + 1
        out_w = (w - size + 2 * padding) // stride + 1

        # 4. 출력을 저장할 배열 초기화
        output = np.zeros((bs, nc, out_h, out_w))

        # 5. Max Pooling 수행
        if ptype == 'max':
            for b in range(bs):  # 각 배치에 대해
                for c in range(nc):  # 각 채널에 대해
                    for i in range(out_h):  # 출력 높이
                        for j in range(out_w):  # 출력 너비
                            h_start = i * stride
                            h_end = h_start + size
                            w_start = j * stride
                            w_end = w_start + size
                            # 영역에서 최대값 찾기
                            pool_region = padded_img[b, c, h_start:h_end, w_start:w_end]
                            output[b, c, i, j] = np.max(pool_region)

        # 6. Average Pooling 수행
        elif ptype == 'avg':
            for b in range(bs):  # 각 배치에 대해
                for c in range(nc):  # 각 채널에 대해
                    for i in range(out_h):  # 출력 높이
                        for j in range(out_w):  # 출력 너비
                            h_start = i * stride
                            h_end = h_start + size
                            w_start = j * stride
                            w_end = w_start + size
                            # 영역의 평균값 계산
                            pool_region = padded_img[b, c, h_start:h_end, w_start:w_end]
                            output[b, c, i, j] = np.mean(pool_region)

        return output

    def relu(self, x):
        return np.maximum(0, x)

    # TODO: Implement a conv-pool-relu-conv-pool-relu-conv-pool-relu network using the above two methods.
    # The input 'x' should have dimensionality num_channels x 400 x 400.
    def forward(self, x):
        # x의 shape: (num_channels x 400 x 400)

        # 1단계: 첫 번째 conv-pool-relu
        # Convolution 1
        # 필터 생성 (예: 입력채널 수 -> 16개의 특징 맵)
        filters1 = np.random.randn(x.shape[0], 3, 3, 16) * 0.01
        conv1 = self.conv(x, filters1, stride=1, padding=1)

        # Pooling 1
        pool1 = self.pool(conv1, size=2, stride=2, ptype='max')

        # ReLU 1
        relu1 = self.relu(pool1)

        # 2단계: 두 번째 conv-pool-relu
        # Convolution 2
        # 필터 생성 (16 -> 32개의 특징 맵)
        filters2 = np.random.randn(relu1.shape[1], 3, 3, 32) * 0.01
        conv2 = self.conv(relu1, filters2, stride=1, padding=1)

        # Pooling 2
        pool2 = self.pool(conv2, size=2, stride=2, ptype='max')

        # ReLU 2
        relu2 = self.relu(pool2)

        # 3단계: 세 번째 conv-pool-relu
        # Convolution 3
        # 필터 생성 (32 -> 64개의 특징 맵)
        filters3 = np.random.randn(relu2.shape[1], 3, 3, 64) * 0.01
        conv3 = self.conv(relu2, filters3, stride=1, padding=1)

        # Pooling 3
        pool3 = self.pool(conv3, size=2, stride=2, ptype='max')

        # ReLU 3
        relu3 = self.relu(pool3)

        return relu3

# Below is just a test code for your reference.
# This is one way to use the conv() and pool() methods above
def main(imfile='test.jpg', outname='output'):
    bs = 16  # Minibatch size
    size = 400  # Input image size (w, h)
    filter_size = 5  # Choose your preferred size
    num_out_maps = 10  # Number of output feature maps.
    num_in_maps = 3
    img = cv2.imread(imfile)  # Returns a numpy ndarray of shape (w, h, num_channel)
    img = cv2.resize(img, (size, size))
    img = np.transpose(img, (2, 0, 1))  # Permute the dimensions so the image shape is 3x400x400
    filters = nr.rand(num_in_maps, filter_size, filter_size, num_out_maps)
    net = MyCNN()
    conv_out = net.conv(img, filters)
    # TODO: Make 'net.forward(img)' work


if __name__ == '__main__':
    main()
