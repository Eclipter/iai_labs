import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def train_net(image_file, m=5, n=5, p=50, e=0.2, alpha=0.001, max_iter=2500):
    np.random.seed(1)

    filtered_filename = "".join([c for c in image_file if c.isalnum()])
    execution_id = f'file={filtered_filename}_m={m}_n={n}_p={p}'

    print(execution_id)

    image = mpimg.imread(image_file)
    height, width, S = image.shape

    if image.max() <= 2.0:
        image *= 255.0

    normalized_img = image.astype(np.float32) * 2.0 / 255.0 - 1.0
    blocks = block_image(normalized_img, m, n)

    N = m * n * S  # vector length
    L = blocks.shape[0]  # blocks count
    Z = (N * L) / ((N + L) * p + 2)  # zip coefficient

    np.random.shuffle(blocks)

    error = math.inf
    W = np.random.uniform(-1, 1, (N, p))
    W_ = W.copy().transpose()

    print(f"Blocks numb: {L} Compression: {Z} Target Error: {e:010.6f}")

    iteration = 0
    while error > e and iteration <= max_iter:
        iteration += 1
        for i, X in enumerate(blocks):
            Y = X @ W
            X_ = Y @ W_

            alpha_ = alpha

            # alpha = 1 / np.sum(X_ * X_)
            # alpha_ = 1 / np.sum(Y * Y)

            delta_X = X_ - X

            W_ = W_ - alpha_ * (Y[:, np.newaxis] @ delta_X[np.newaxis, :])
            W = W - alpha * (X_[:, np.newaxis] @ delta_X[np.newaxis, :] @ W_.T)

            W = W / np.apply_along_axis(lambda col: math.sqrt(np.sum(col * col)), axis=0, arr=W)
            W_ = W_ / np.apply_along_axis(lambda col: math.sqrt(np.sum(col * col)), axis=0, arr=W_)

        error = 0
        for X in blocks:
            X_ = X @ W @ W_
            delta_X = X_ - X
            partial_e = np.sum(np.power(delta_X, 2)) / 2
            error += partial_e
        error /= L
        print(f"Iteration: {iteration} Total Error:{error:010.6f}")
    np.save(os.path.join('weights', f'weights_{execution_id}'), W)
    np.save(os.path.join('weights', f'weights_back_{execution_id}'), W_)
    return W, W_, execution_id, iteration, error, L, Z


def block_image(img, m, n):
    img = img.copy()
    h, w = img.shape[:2]

    last_orig_block_end_w, overlay_block_start_w = count_block_borders(w, n)
    last_orig_block_end_h, overlay_block_start_h = count_block_borders(h, m)

    img = np.hstack((img[:, :last_orig_block_end_w], img[:, overlay_block_start_w:]))
    img = np.vstack((img[:last_orig_block_end_h, :], img[overlay_block_start_h:]))
    h, w, d = img.shape

    return img.reshape(h // m, m, w // n, n, d).swapaxes(1, 2).reshape(-1, m, n, d).reshape(-1, m * n * d)


def restore_image(blocks, m, n, s, h, w):
    blocks = blocks.copy()

    blocks.shape = (-1, m, n, s)
    last_orig_block_end_w, overlay_block_start_w = count_block_borders(w, n)
    last_orig_block_end_h, overlay_block_start_h = count_block_borders(h, m)

    h_blocks_count = (last_orig_block_end_h + m) // m
    w_blocks_count = (last_orig_block_end_w + n) // n

    blocks.shape = (h_blocks_count, w_blocks_count, m, n, s)
    blocks = blocks.swapaxes(1, 2).reshape(h_blocks_count * m, -1, n, s).reshape(h_blocks_count * m, -1, s)
    blocks = np.hstack((blocks[:, :overlay_block_start_w], blocks[:, -n:]))
    blocks = np.vstack((blocks[:overlay_block_start_h], blocks[-m:]))
    return blocks


def compress_image(img, W, m, n):
    img_normalized = img.astype(np.float32) * 2.0 / 255.0 - 1.0
    blocks = block_image(img_normalized, m, n)
    return np.apply_along_axis(lambda block: block @ W, axis=1, arr=blocks)


def uncompress_image(blocks, W_, m, n, s, h, w):
    f_blocks = np.apply_along_axis(lambda block: block @ W_, axis=1, arr=blocks)
    img = restore_image(f_blocks, m, n, s, h, w)
    img = np.clip(img, -1.0, 1.0)
    value_range = img.max() - img.min()
    return ((img - img.min()) / value_range * 255.0).astype(np.uint8)


def count_block_borders(img_length, block_length):
    return img_length - img_length % block_length, img_length - block_length


def draw_images(original, restored, file_name='comparison'):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(original)
    ax1.set_title('Original')

    ax2.imshow(restored)
    ax2.set_title('Restored')

    plt.savefig(f'comp_{file_name}')
    plt.show()


def train_and_show(image_file, W, W_, m, n):
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
    alpha = 0.001
    p_seq = [30 + x * 5 for x in range(10)]
    results = []
    for p in p_seq:
        W, W_, run_id, iteration, error, L, Z = train_net('images/drones_150x150.jpg', m, n, p, e, alpha)
        results.append((p, Z, iteration))

    plt.plot([r[1] for r in results], [r[2] for r in results], color='r', marker='.', linestyle='None')
    plt.xlabel("Compression Coefficient")
    plt.ylabel("Iterations")
    plt.savefig('report_compression')
    plt.show()
    return results


def test_iterations_on_alpha():
    e = 0.2
    m = 5
    n = 5
    p = 50
    alpha_seq = [0.0005 + 0.0001 * x for x in range(5)]
    results = []
    for alpha in alpha_seq:
        print("Alpha: " + str(alpha))
        W, W_, run_id, iteration, error, L, Z = train_net('images/sim_ther_200x200.jpg', m, n, p, e, alpha)
        results.append((alpha, iteration))

    plt.plot([r[0] for r in results], [r[1] for r in results], color='r', marker='.', linestyle='None')
    plt.xlabel("Learning Rate")
    plt.ylabel("Iterations")
    plt.savefig('report_rate')
    plt.show()
    return results


def test_iterations_on_files():
    e = 0.2
    m = 5
    n = 5
    p = 50
    alpha = 0.001
    images = [
        'images/chrome_100x100.jpg',
        'images/drones_150x150.jpg',
        'images/sim_ther_200x200.jpg',
    ]
    results = []
    for img in images:
        W, W_, run_id, iteration, error, L, Z = train_net(img, m, n, p, e, alpha)
        train_and_show(img, W, W_, m, n)
        results.append((img, iteration))
    return results


def test_iterations_on_error():
    m = 5
    n = 5
    p = 50
    alpha = 0.001
    e_seq = [0.1 + 0.05 * x for x in range(5)]
    results = []
    for e in e_seq:
        W, W_, run_id, iteration, error, L, Z = train_net('images/sim_ther_200x200.jpg', m, n, p, e, alpha)
        results.append((e, iteration))

    plt.plot([r[0] for r in results], [r[1] for r in results], color='r', marker='.', linestyle='None')
    plt.xlabel("Target Error")
    plt.ylabel("Iterations")
    plt.savefig('report_error')
    plt.show()
    return results


def main():
    # print(test_iterations_on_neurons())
    # print(test_iterations_on_files())
    # print(test_iterations_on_alpha())
    print(test_iterations_on_error())

    file = 'images/drones_150x150.jpg'
    m = 5
    n = 5
    p = 50
    e = 0.2
    alpha = 0.001
    W, W_, execution_id, iteration, error, L, Z = train_net(file, m, n, p, e, alpha)
    train_and_show(file, W, W_, m, n)


if __name__ == '__main__':
    main()