import os



def LDA_trainer(m, dest_dir, prefix, n_batch=10, batch_size=100):
    """
    Takes `m`, an LDAGibbs instance; `dest_dir`, a string specifying
    the destination directory for periodically saving the instance;
    `prefix`, the filename prefix; `n_batch `...
    """
    for i in xrange(n_batch):

        m.train(itr=batch_size)

        filename = '{0}-{1}.npz'.format(prefix, m.iterations)

        filename = os.path.join(dest_dir, filename)

        m.save(filename)
