from IPython.display import clear_output

import os, shutil, requests, sys, time, logging

logging.basicConfig(filename='output.log', level=logging.INFO)

def print_console_log(str):
    print str
    logging.info(str)

def get_batch(X, y, current_batch_index, batch_size):
    batch_start_index = current_batch_index * batch_size
    batch_end_index = (current_batch_index + 1) * batch_size
    batch_features = X.iloc[batch_start_index:batch_end_index].values
    batch_labels = y.iloc[batch_start_index:batch_end_index].values
    return batch_features, batch_labels

def print_stats(session, epoch, train_features, train_labels, valid_features, 
                valid_labels, loss, accuracy, x_tensor, y_tensor, keep_prob_tensor, 
                summary_op, train_summary_writer, test_summary_writer):
    feed_loss = { x_tensor: train_features, y_tensor: train_labels, keep_prob_tensor: 1.0 }
    feed_valid = { x_tensor: valid_features, y_tensor: valid_labels, keep_prob_tensor: 1.0 }
    loss = session.run(loss, feed_loss)
    accuracy = session.run(accuracy, feed_valid)
    train_summary_writer.add_summary(session.run(summary_op, feed_loss), epoch)
    train_summary_writer.flush()
    test_summary_writer.add_summary(session.run(summary_op, feed_valid), epoch)
    test_summary_writer.flush()
    print_console_log('Epoch {:>2}: '.format(epoch + 1) + 'loss: %.4f' % loss + ', accuracy: %.3f' % accuracy)
    return loss, accuracy
    
def clear_model(save_model_path, tensorboard_path):
    if os.path.exists(save_model_path):
        shutil.rmtree(save_model_path)
    if os.path.exists(tensorboard_path):
        shutil.rmtree(tensorboard_path)

def download_file(url) :
    localFilename = url.split('/')[-1]
    with open(localFilename, 'wb') as f:
        start = time.clock()
        r = requests.get(url, stream=True)
        total_length = int(r.headers.get('content-length'))
        dl = 0
        if total_length is None: # no content length header
            f.write(r.content)
        else:
            for chunk in r.iter_content(1024):
                dl += len(chunk)
                f.write(chunk)
                done = int(50 * dl / total_length)
                clear_output(wait=True)
                print "\r[%s%s] %s bps" % ('=' * done, ' ' * (50 - done), 
                    dl // (time.clock() - start))
                print ''
    return (time.clock() - start)