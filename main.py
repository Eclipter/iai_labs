import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def train_net(image_file, m=5, n=5, p=50, e=1., max_iter=2500):
    np.random.seed(1)

    filtered_filename = "".join([c for c in image_file if c.isalnum()])
    run_id = f'file={filtered_filename}_m={m}_n={n}_p={p}'

    print(run_id)

    image = mpimg.imread(image_file)
    height, width, S = image.shape

    if image.max() <= 2.0:
        image *= 255.0

    normalized_img = image.astype(np.float32) * 2.0 / 255.0 - 1.0
    blocks = block_image(normalized_img, m, n)

    N = m * n * S
    L = blocks.shape[0]
    Z = (N * L) / ((N + L) * p + 2)

    np.random.shuffle(blocks)

    # e = 0.001 * m * n * p
    error = math.inf
    W = np.random.uniform(-1, 1, (N, p))
    W_ = W.copy().transpose()

    print(f"Blocks numb: {blocks.shape[0]} Target Error: {e:010.6f}")

    iteration = 0
    while error > e and iteration <= max_iter:
        iteration += 1
        for i, X in enumerate(blocks):
            Y = X @ W
            X_ = Y @ W_

            alpha = 0.005
            alpha_ = 0.005

            # alpha = 1 / np.sum(X_ * X_)
            # alpha_ = 1 / np.sum(Y * Y)

            delta_X = X_ - X

            W_ = W_ - alpha_ * (Y[:, np.newaxis] @ delta_X[np.newaxis, :])
            W = W - alpha * (X_[:, np.newaxis] @ delta_X[np.newaxis, :] @ W_.T)

            W = W / np.apply_along_axis(lambda col: math.sqrt(np.sum(col * col)), axis=0, arr=W)
            W_ = W_ / np.apply_along_axis(lambda col: math.sqrt(np.sum(col * col)), axis=0, arr=W_)

        # error = 0
        # for X in blocks:
        #     X_ = X @ W @ W_
        #     delta_X = X_ - X
        #     partial_e = np.sum(np.power(delta_X, 2)) / 2
        #     error += partial_e
        # error /= L
        error = np.sum(np.apply_along_axis(lambda X: np.sum(np.power((X @ W @ W_) - X, 2)) / 2.0, axis=1, arr=blocks)) / L
        print(f"Iteration: {iteration} Total Error:{error:010.6f}")
    np.save(os.path.join('weights', f'weights_{run_id}'), W)
    np.save(os.path.join('weights', f'weights_back_{run_id}'), W_)
    return W, W_, run_id, iteration, error, L, Z


def block_image(img, m, n):
    img = img.copy()
    h, w = img.shape[:2]

    pivot_w, start_w = count_block_borders(w, n)
    pivot_h, start_h = count_block_borders(h, m)

    img = np.hstack((img[:, :pivot_w], img[:, start_w:]))
    img = np.vstack((img[:pivot_h, :], img[start_h:]))
    h, w, d = img.shape

    return img.reshape(h // m, m, w // n, n, d).swapaxes(1, 2).reshape(-1, m, n, d).reshape(-1, m * n * d)


def restore_image(blocks, m, n, s, h, w):
    blocks = blocks.copy()

    blocks.shape = (-1, m, n, s)
    pivot_w, start_w = count_block_borders(w, n)
    pivot_h, start_h = count_block_borders(h, m)

    h_blocks_num = (pivot_h + m) // m
    w_blocks_num = (pivot_w + n) // n

    blocks.shape = (h_blocks_num, w_blocks_num, m, n, s)
    blocks = blocks.swapaxes(1, 2).reshape(h_blocks_num * m, -1, n, s).reshape(h_blocks_num * m, -1, s)
    blocks = np.hstack((blocks[:, :start_w], blocks[:, -n:]))
    blocks = np.vstack((blocks[:start_h], blocks[-m:]))
    return blocks


def compress_image(img, W, m, n):
    normalized_img = img.astype(np.float32) * 2.0 / 255.0 - 1.0
    blocks = block_image(normalized_img, m, n)
    return np.apply_along_axis(lambda block: block @ W, axis=1, arr=blocks)


def uncompress_image(blocks, W_, m, n, s, h, w):
    f_blocks = np.apply_along_axis(lambda block: block @ W_, axis=1, arr=blocks)
    img = restore_image(f_blocks, m, n, s, h, w)
    img = np.clip(img, -1.0, 1.0)
    value_range = img.max() - img.min()
    return ((img - img.min()) / value_range * 255.0).astype(np.uint8)


def count_block_borders(full_l, block_l):
    return full_l - full_l % block_l, full_l - block_l


def draw_images(original, restored, file_name='comparison'):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(original)
    ax1.set_title('Original')

    ax2.imshow(restored)
    ax2.set_title('Restored')

    plt.savefig(f'comp_{file_name}')
    plt.show()


def demo_on_image(image_file, W, W_, m, n):
    image = mpimg.imread(image_file)
    image_height, image_width, s = image.shape
    compressed_image = compress_image(image, W, m, n)
    uncompressed_image = uncompress_image(compressed_image, W_, m, n, s, image_height, image_width)
    clean_file_name = image_file[image_file.rfind("/") + 1:image_file.rfind(".")]
    mpimg.imsave(f'output/{clean_file_name}', uncompressed_image)
    draw_images(image, uncompressed_image, clean_file_name)


def test_iterations_on_neurons():
    e = 0.25
    m = 5
    n = 5
    p_seq = [40 + x * 5 for x in range(5)]
    results = []
    for p in p_seq:
        W, W_, run_id, iteration, error, L, Z = train_net('images/download.jpeg', m, n, p, e)
        results.append((p, Z, iteration))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title('Compression')
    ax1.plot([r[0] for r in results], [r[1] for r in results], color='r', marker='.')
    ax2.set_title('Iterations')
    ax2.plot([r[0] for r in results], [r[2] for r in results], marker='.')
    plt.savefig('report_1')
    plt.show()
    return results


def test_iterations_on_files():
    e = 0.25
    m = 5
    n = 5
    p = 50
    images = [
        'images/test256.jpg',
        'images/barcode185x180.jpeg',
        'images/firefox128.jpg',
        'images/dog100.bmp'
    ]
    results = []
    for img in images:
        W, W_, run_id, iteration, error, L, Z = train_net(img, m, n, p, e)
        demo_on_image(img, W, W_, m, n)
        results.append((img, iteration))
    return results


def test_iterations_to_error():
    m = 5
    n = 5
    p = 50
    e_seq = [0.2 + 0.05 * x for x in range(5)]
    results = []
    for e in e_seq:
        W, W_, run_id, iteration, error, L, Z = train_net('images/download.jpeg', m, n, p, e)
        results.append((e, iteration))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title('Compression')
    ax1.plot([r[0] for r in results], [r[1] for r in results], color='r', marker='.')
    ax2.set_title('Iterations')
    ax2.plot([r[0] for r in results], [r[2] for r in results], marker='.')
    plt.savefig('report_1')
    plt.show()
    return results


def main():
    # print(test_iterations_on_neurons())
    # print(test_iterations_on_files())

    file = 'images/pic200x200.jpg'
    m = 5
    n = 5
    p = 50
    e = 0.25
    W, W_, run_id, iteration, error, L, Z = train_net(file, m, n, p, e)
    demo_on_image(file, W, W_, m, n)


if __name__ == '__main__':
    main()