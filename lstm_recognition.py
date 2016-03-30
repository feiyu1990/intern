'''
Build a tweet sentiment analyzer
'''

from __future__ import print_function
import cPickle as pickle

from collections import OrderedDict, defaultdict
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import random
import imdb



correct_list = {'5_19479358@N00':'Museum', '38_59616483@N00':'Museum','136_95413346@N00':'Museum',
                    '0_27302158@N00':'CasualFamilyGather','7_55455788@N00':'Birthday',
                    '144_95413346@N00':'Halloween', '29_13125640@N07':'Christmas', '1_21856707@N00': 'GroupActivity',
                    '0_22928590@N00':'GroupActivity','3_7531619@N05':'Zoo',
                    '16_18108851@N00':'Show', '23_89182227@N00':'Show', '2_27883710@N08':'Sports',
                    '35_8743691@N02':'Wedding', '14_93241698@N00':'Museum', '9_34507951@N07':'BusinessActivity',
                    '32_35578067@N00':'Protest', '20_89138584@N00':'PersonalSports', '18_50938313@N00':'PersonalSports',
                    '376_86383385@N00':'PersonalSports','439_86383385@N00':'PersonalSports','545_86383385@N00':'PersonalSports',
                    '2_43198495@N05':'PersonalSports', '3_60652642@N00':'ReligiousActivity', '9_60053005@N00':'GroupActivity',

                        '56_74814994@N00':'BusinessActivity', '22_32994285@N00':'Sports', '15_66390637@N08':'Sports',
                         '3_54218473@N05':'Zoo', '4_53628484@N00':'Sports', '0_7706183@N06':'GroupActivity',
                         '4_15251430@N03':'Zoo', '63_52304204@N00':'Sports', '2_36319742@N05':'Architecture',
                         '2_12882543@N00':'Sports', '1_75003318@N00':'Sports', '1_88464035@N00':'GroupActivity',
                         '21_49503048699@N01':'CasualFamilyGather', '211_86383385@N00':'Sports',
                         '0_70073383@N00':'PersonalArtActivity'}



dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}

root = '/home/ubuntu/lstm/data_new/'
# datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


# def get_dataset(name):
#     return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    params['Wemb'] = (numpy.load(root + web_path)).astype(config.floatX)
    # print(params['Wemb'])
    # params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk == 'Wemb':
            continue
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]

layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update



def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj_sequence = proj * mask[:, :, None]
    if options['use_dropout']:
        proj_sequence = dropout_layer(proj_sequence, use_noise, trng)
    pred_img = tensor.tensordot(proj_sequence, tparams['U'], axes=[[2], [0]]) + tparams['b']
    pred_img = pred_img.dimshuffle([1,0,2])
    # pred_img = tensor.DimShuffle(pred_img, (1, 0, 2))
    pred_img = tensor.reshape(pred_img, (pred_img.shape[0] * pred_img.shape[1], pred_img.shape[2]))
    pred_img = tensor.nnet.softmax(pred_img)
    # mask_reshape = tensor.DimShuffle(mask, (1, 0))
    mask_reshape = mask.copy()
    mask_reshape = mask_reshape.dimshuffle([1, 0])
    mask_reshape = tensor.reshape(mask_reshape, (mask_reshape.shape[0] * mask_reshape.shape[1], 1))
    pred_img = pred_img * mask_reshape
    # (99, 64, 128) (128, 23) -> (99, 64, 23)

    f_pred_h_sequence = theano.function([x, mask], pred_img, name='f_pred_h_sequence')

    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost, f_pred_h_sequence

def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 23)).astype(config.floatX)
    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        # print('x', x.shape, x)
        # print('mask', mask.shape, mask)
        # print('y', y.shape,  y)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

def pred_error_softtarget(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.argmax(y, axis=1)
        # targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err


def pred_h_sequence(f_pred_h_sequence, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    # n_samples = len(data[0])
    n_done = 0
    probs = numpy.zeros((50000, 23)).astype(config.floatX)
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_prob = f_pred_h_sequence(x, mask)
        print(pred_prob)
        probs[n_done:n_done + pred_prob.shape[0], :] = pred_prob
        print(n_done)
        n_done += pred_prob.shape[0]
    probs = probs[:n_done, :]
    print(probs.shape)
        # if verbose:
        #     print('%d/%d samples classified' % (n_done, n_samples))

    return probs

def train_lstm(
    training_data_path,test_data_path, validation_data_path,
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.01,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    # n_words=10000,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    # optimizer=rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model',  # The best model will be saved there
    validFreq=-1,  # Compute the validation error after this number of update.
    saveFreq=-1,  # Save the parameters after every saveFreq updates
    maxlen=1000,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    # load_data, prepare_data = get_dataset(dataset)
    prepare_data = imdb.prepare_data

    print('Loading data')
    with open(root + training_data_path) as f:
        train = pickle.load(f)
    with open(root + test_data_path) as f:
        test = pickle.load(f)
    with open(root + validation_data_path) as f:
        valid = pickle.load(f)
    valid_sample_idx = random.sample(range(len(valid[0])), 50)
    valid = ([valid[0][i] for i in valid_sample_idx], [valid[1][i] for i in valid_sample_idx])
    # temp = range(len(train_[0]))
    # random.shuffle(temp)
    # valid_portion = 0.05
    # valid_ind = temp[:int(valid_portion*len(temp))]
    # train_ind = temp[int(valid_portion*len(temp)):]
    # print(valid_ind, train_ind)
    # train = ([train_[0][i] for i in train_ind], [train_[1][i] for i in train_ind])
    # valid = ([train_[0][i] for i in valid_ind], [train_[1][i] for i in valid_ind])


    if test_size > 0:
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    ydim = numpy.max(train[1]) + 1
    model_options['ydim'] = ydim

    print('Building model')
    params = init_params(model_options)

    if reload_model:
        load_params(reload_model, params)

    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost, _) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                # if saveto and numpy.mod(uidx, saveFreq) == 0:
                #     print('Saving...')
                #
                #     if best_p is not None:
                #         params = best_p
                #     else:
                #         params = unzip(tparams)
                #     numpy.savez(saveto + '_' + str(uidx) + '_alltraining.npz', history_errs=history_errs, **params)
                #     pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                #     print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    # print('train_err')
                    # train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid,
                                           kf_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):
                        best_p = unzip(tparams)
                        bad_counter = 0

                    # print( ('Train ', train_err, 'Valid ', valid_err,
                    #        'Test ', test_err) )

                    print( ('Valid ', valid_err,
                           'Test ', test_err) )
                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    if saveto:
        numpy.savez(saveto + '_alltraining.npz', train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err

#last output
def build_model_last_output(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')
    mask_last_output = tensor.matrix('mask_last_output', dtype=config.floatX)
    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj_sequence = proj * mask[:, :, None]
    if options['use_dropout']:
        proj_sequence = dropout_layer(proj_sequence, use_noise, trng)
    pred_img = tensor.tensordot(proj_sequence, tparams['U'], axes=[[2], [0]]) + tparams['b']
    pred_img = pred_img.dimshuffle([1,0,2])
    # pred_img = tensor.DimShuffle(pred_img, (1, 0, 2))
    pred_img = tensor.reshape(pred_img, (pred_img.shape[0] * pred_img.shape[1], pred_img.shape[2]))
    pred_img = tensor.nnet.softmax(pred_img)
    mask_reshape = mask.copy()
    mask_reshape = mask_reshape.dimshuffle([1, 0])
    mask_reshape = tensor.reshape(mask_reshape, (mask_reshape.shape[0] * mask_reshape.shape[1], 1))
    pred_img = pred_img * mask_reshape
    f_pred_h_sequence = theano.function([x, mask], pred_img, name='f_pred_h_sequence')



    if options['encoder'] == 'lstm':
        proj = (proj * mask_last_output[:, :, None]).sum(axis=0)
        proj = proj / mask_last_output.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    f_pred_prob = theano.function([x, mask, mask_last_output], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask, mask_last_output], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, mask_last_output, y, f_pred_prob, f_pred, cost, f_pred_h_sequence
def adadelta_last_output(lr, tparams, grads, x, mask, mask_last_output, y, cost):

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, mask_last_output, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update
def pred_error_last_output(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, mask_last_output, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask, mask_last_output)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err
def train_lstm_last_output(
    training_data_path,test_data_path, validation_data_path,
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.01,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    # n_words=10000,  # Vocabulary size
    optimizer=adadelta_last_output,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    # optimizer=rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model',  # The best model will be saved there
    validFreq=-1,  # Compute the validation error after this number of update.
    saveFreq=-1,  # Save the parameters after every saveFreq updates
    maxlen=1000,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    # load_data, prepare_data = get_dataset(dataset)
    prepare_data = imdb.prepare_data_last_output

    print('Loading data')
    with open(root + training_data_path) as f:
        train = pickle.load(f)
    with open(root + test_data_path) as f:
        test = pickle.load(f)
    with open(root + validation_data_path) as f:
        valid = pickle.load(f)
    valid_sample_idx = random.sample(range(len(valid[0])), 50)
    valid = ([valid[0][i] for i in valid_sample_idx], [valid[1][i] for i in valid_sample_idx])
    # temp = range(len(train_[0]))
    # random.shuffle(temp)
    # valid_portion = 0.05
    # valid_ind = temp[:int(valid_portion*len(temp))]
    # train_ind = temp[int(valid_portion*len(temp)):]
    # print(valid_ind, train_ind)
    # train = ([train_[0][i] for i in train_ind], [train_[1][i] for i in train_ind])
    # valid = ([train_[0][i] for i in valid_ind], [train_[1][i] for i in valid_ind])


    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    ydim = numpy.max(train[1]) + 1
    model_options['ydim'] = ydim

    print('Building model')
    params = init_params(model_options)

    if reload_model:
        load_params(reload_model, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, mask_last_output,
     y, f_pred_prob, f_pred, cost, _) = build_model_last_output(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, mask_last_output, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, mask_last_output, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, mask_last_output, y, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, mask_last_output, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, mask_last_output, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto + '_' + str(uidx) + '_alltraining.npz', history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    # print('train_err')
                    # train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error_last_output(f_pred, prepare_data, valid,
                                           kf_valid)
                    test_err = pred_error_last_output(f_pred, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    # print( ('Train ', train_err, 'Valid ', valid_err,
                    #        'Test ', test_err) )

                    print( ('Valid ', valid_err,
                           'Test ', test_err) )
                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error_last_output(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error_last_output(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error_last_output(f_pred, prepare_data, test, kf_test)

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    if saveto:
        numpy.savez(saveto + '_alltraining.npz', train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err

def prepare_data_softtarget(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    y = numpy.array(labels).astype('float32')
    # print(y.shape)

    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, y


def build_model_softtarget(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.matrix('y', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj_sequence = proj * mask[:, :, None]
    if options['use_dropout']:
        proj_sequence = dropout_layer(proj_sequence, use_noise, trng)
    pred_img = tensor.tensordot(proj_sequence, tparams['U'], axes=[[2], [0]]) + tparams['b']
    pred_img = pred_img.dimshuffle([1,0,2])
    # pred_img = tensor.DimShuffle(pred_img, (1, 0, 2))
    pred_img = tensor.reshape(pred_img, (pred_img.shape[0] * pred_img.shape[1], pred_img.shape[2]))
    pred_img = tensor.nnet.softmax(pred_img)
    # mask_reshape = tensor.DimShuffle(mask, (1, 0))
    mask_reshape = mask.copy()
    mask_reshape = mask_reshape.dimshuffle([1, 0])
    mask_reshape = tensor.reshape(mask_reshape, (mask_reshape.shape[0] * mask_reshape.shape[1], 1))
    pred_img = pred_img * mask_reshape
    # (99, 64, 128) (128, 23) -> (99, 64, 23)

    f_pred_h_sequence = theano.function([x, mask], pred_img, name='f_pred_h_sequence')

    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.sum(y * tensor.log(pred+ off)) / n_samples

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost, f_pred_h_sequence

def train_lstm_softtarget(
    training_data_path,test_data_path, validation_data_path,
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.01,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    # n_words=10000,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    # optimizer=rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model',  # The best model will be saved there
    validFreq=-1,  # Compute the validation error after this number of update.
    saveFreq=-1,  # Save the parameters after every saveFreq updates
    maxlen=1000,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    # load_data, prepare_data = get_dataset(dataset)
    prepare_data = prepare_data_softtarget

    print('Loading data')
    with open(root + training_data_path) as f:
        train = pickle.load(f)
    with open(root + test_data_path) as f:
        test = pickle.load(f)
    with open(root + validation_data_path) as f:
        valid = pickle.load(f)

    # valid_sample_idx = random.sample(range(len(valid[0])), 50)
    # valid = ([valid[0][i] for i in valid_sample_idx], [valid[1][i] for i in valid_sample_idx])
    # temp = range(len(train_[0]))
    # random.shuffle(temp)
    # valid_portion = 0.05
    # valid_ind = temp[:int(valid_portion*len(temp))]
    # train_ind = temp[int(valid_portion*len(temp)):]
    # print(valid_ind, train_ind)
    # train = ([train_[0][i] for i in train_ind], [train_[1][i] for i in train_ind])
    # valid = ([train_[0][i] for i in valid_ind], [train_[1][i] for i in valid_ind])


    if test_size > 0:
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    ydim = 23
    model_options['ydim'] = ydim

    print('Building model')
    params = init_params(model_options)

    if reload_model:
        load_params(reload_model, params)

    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost, _) = build_model_softtarget(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                # if saveto and numpy.mod(uidx, saveFreq) == 0:
                #     print('Saving...')
                #
                #     if best_p is not None:
                #         params = best_p
                #     else:
                #         params = unzip(tparams)
                #     numpy.savez(saveto + '_' + str(uidx) + '_alltraining.npz', history_errs=history_errs, **params)
                #     pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                #     print('Done')

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    # print('train_err')
                    # train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error_softtarget(f_pred, prepare_data, valid,
                                           kf_valid)
                    test_err = pred_error_softtarget(f_pred, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    # print( ('Train ', train_err, 'Valid ', valid_err,
                    #        'Test ', test_err) )

                    print( ('Valid ', valid_err,
                           'Test ', test_err) )
                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error_softtarget(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error_softtarget(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error_softtarget(f_pred, prepare_data, test, kf_test)

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    if saveto:
        numpy.savez(saveto + '_alltraining.npz', train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err



def test_prediction(
    load_name,
    save_name,
    feature_path,
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model',  # The best model will be saved there
    validFreq=-1,  # Compute the validation error after this number of update.
    saveFreq=-1,  # Save the parameters after every saveFreq updates
    maxlen=1000,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    test_size=-1,
):
    model_options = locals().copy()
    # with open(root + 'training_imdb.pkl') as f:
    #     train = pickle.load(f)
    with open(root + feature_path) as f:
        test = pickle.load(f)
    ydim = 23
    # print(ydim)
    model_options['ydim'] = ydim

    print('Building model')
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)
    load_params(root + '../model_good/' + load_name, params)

    prepare_data = imdb.prepare_data
    tparams = init_tparams(params)
    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost, f_pred_h_sequence) = build_model(tparams, model_options)

    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)
    test_prob = pred_probs(f_pred_prob, prepare_data, test, kf_test)
    test_error = pred_error(f_pred, prepare_data, test, kf_test)
    print(test_error)
    print(test_prob.shape, numpy.argmax(test_prob, axis=1)[:100])
    # print(root + save_name + '.npy')
    numpy.save(root + save_name + '.npy', test_prob)

def lstm_recognition(list_name, name, test_name):
    prediction = numpy.load(root + name + '.npy')
    event_prediction_dict = dict()
    with open(root + 'pec_'+list_name+'_fc7_vote_multilabel_event_recognition_expand_balanced_3_iter_100000_event_list.pkl') as f:
    # with open(root + list_name + '_list.pkl') as f:
        test_event_list = pickle.load(f)
    with open(root + test_name) as f:
        _, test_label = pickle.load(f)

    count = 0
    for event_id in test_event_list:
        event_prediction_dict[event_id] = prediction[count, :]
        count += 1
    with open(root + name + '_prediction_dict.pkl', 'w') as f:
        pickle.dump(event_prediction_dict, f)


    print(name, numpy.mean(numpy.argmax(prediction, axis=1) == test_label))


def lstm_recognition_crossvalidation(name, test_name,fold=5):
    event_prediction_dict = dict()

    for i in xrange(fold):
        with open(root + name + '_' + str(i) + '_prediction_dict.pkl') as f:
            temp = pickle.load(f)
        for event_id in temp:
            if event_id in event_prediction_dict:
                event_prediction_dict[event_id] = event_prediction_dict[event_id] + numpy.array(temp[event_id])
            else:
                event_prediction_dict[event_id] = numpy.array(temp[event_id])
    for event_id in event_prediction_dict:
        event_prediction_dict[event_id] /= fold


    with open(root + name + '_crossvalidation_prediction_dict.pkl', 'w') as f:
        pickle.dump(event_prediction_dict, f)

def lstm_img_create_dict():
    test_prob = numpy.load(root + 'test_lstm_prediction_img.npy')
    event_img_dict = defaultdict(list)
    for event_name in dict_name2:
        f = open('/home/ubuntu/event_curation/baseline_all_0509/' + event_name + '/test_image_ids.cPickle')
        temp = pickle.load(f)
        f.close()
        for i in temp:
            event_img_dict[i.split('/')[0]].append(i)
    with open(root + 'test_event_list.pkl') as f:
        event_list = pickle.load(f)
    img_prediction_dict = dict()
    count = 0
    for inx, event in enumerate(event_list):
        temp_ = event_img_dict[event]
        print(inx, event, count, len(temp_))
        for i in temp_:
            if numpy.abs(numpy.sum(test_prob[count, :]) - 1) > 0.01:
                print(count, 'ERROR!', numpy.sum(test_prob[count, :]))
            img_prediction_dict[i] = test_prob[count, :]
            count += 1
        while numpy.abs(numpy.sum(test_prob[count, :]) - 1) > 0.01 and count < test_prob.shape[0] - 1:
            count += 1
    f = open(root + 'test_lstm_prediction_img_dict.pkl', 'w')
    pickle.dump(img_prediction_dict, f)
    f.close()

if __name__ == '__main__':

    # See function train for all possible parameter and there definition.
    # train_lstm(
    #     training_data_path='multilabel_iter1w_training_imdb_oversample_20_0.5.pkl',
    #     test_data_path='multilabel_iter1w_test_imdb.pkl',
    #     validation_data_path='multilabel_iter1w_test_imdb.pkl',
    #     max_epochs=100,
    #     # reload_model = '../model_good/lstm_model_522_alltraining.npz',
    #     validFreq=100,
    #     saveFreq=100,
    #     saveto='multilabel_iter1w_oversample_20_0.5',
    #     patience=20
    #     # test_size=500,
    # )
    # web_path = 'multilabel_feature_all.npy'
    # train_lstm(
    #     training_data_path='vote_multilabel_training.pkl',
    #     test_data_path='vote_multilabel_test.pkl',
    #     validation_data_path='vote_multilabel_validation.pkl',
    #     max_epochs=35,
    #     # reload_model = '../model_good/lstm_model_522_alltraining.npz',
    #     validFreq=100,
    #     saveFreq=100,
    #     saveto='/home/ubuntu/lstm/model_good/vote_multilabel',
    #     patience=10
    #     # test_size=500,
    # )

    # web_path = 'soft_multilabel_feature_test_pca.npy'
    # test_prediction('vote_softall_multilabel_alltraining.npz', 'soft_vote_multilabel_test_all', 'vote_multilabel_test_all.pkl')
    # lstm_recognition('test', 'soft_vote_multilabel_test_all', 'vote_multilabel_test_all.pkl')
    # #
    # test_prediction('vote_softall_multilabel_alltraining.npz', 'soft_vote_multilabel_validation', 'vote_multilabel_validation.pkl')
    # lstm_recognition('validation', 'soft_vote_multilabel_validation', 'vote_multilabel_validation.pkl')

    #

    web_path = 'pec_soft_feature_test_pca.npy'
    test_prediction('vote_softall_multilabel_alltraining.npz', 'pec_vote_multilabel_soft_test', 'pec_vote_multilabel_soft_test.pkl')
    lstm_recognition('test', 'pec_vote_multilabel_soft_test', 'pec_vote_multilabel_soft_test.pkl')
    web_path = 'pec_soft_feature_validation_pca.npy'
    test_prediction('vote_softall_multilabel_alltraining.npz', 'pec_vote_multilabel_soft_validation', 'pec_vote_multilabel_soft_validation.pkl')
    lstm_recognition('training', 'pec_vote_multilabel_soft_validation', 'pec_vote_multilabel_soft_validation.pkl')

    web_path = 'pec_feature_test_pca.npy'
    test_prediction('vote_multilabel_alltraining.npz', 'pec_vote_multilabel_test', 'pec_vote_multilabel_test.pkl')
    lstm_recognition('test', 'pec_vote_multilabel_test', 'pec_vote_multilabel_test.pkl')
    web_path = 'pec_feature_validation_pca.npy'
    test_prediction('vote_multilabel_alltraining.npz', 'pec_vote_multilabel_validation', 'pec_vote_multilabel_validation.pkl')
    lstm_recognition('training', 'pec_vote_multilabel_validation', 'pec_vote_multilabel_validation.pkl')

    #


    # web_path = 'soft_multilabel_feature_all.npy'
    # test_prediction('vote_multilabel_alltraining.npz', 'vote_multilabel_test_all', 'vote_multilabel_test_all.pkl')
    # lstm_recognition('vote_multilabel_test_all', 'vote_multilabel_test_all.pkl')

    #
    # print('HI USING SOFT')
    # web_path = 'soft_multilabel_feature_all.npy'
    # train_lstm_softtarget(
    #         # training_data_path='vote_softall_multilabel_training_'+str(i)+'_oversample_20_0.5.pkl',
    #         training_data_path='vote_softall_multilabel_training.pkl',
    #         test_data_path='vote_softall_multilabel_test.pkl',
    #         validation_data_path='vote_softall_multilabel_validation.pkl',
    #         max_epochs=30,
    #         # reload_model = '../model_good/lstm_model_522_alltraining.npz',
    #         validFreq=100,
    #         saveFreq=100,
    #         saveto='/home/ubuntu/lstm/model_good/vote_softall_multilabel',
    #         # saveto='/home/ubuntu/lstm/model_good/vote_softall_multilabel_oversample_20_0.5_' + str(i),
    #         patience=10
    #         # test_size=500,
    # )




    # # #
    # for i in range(5):
    #     web_path = 'vote_multilabel_feature_all.npy'
    #     train_lstm(
    #         training_data_path='vote_multilabel_training_'+str(i)+'_oversample_20_0.5.pkl',
    #         # training_data_path='vote_multilabel_training_'+str(i)+'.pkl',
    #         test_data_path='vote_multilabel_test.pkl',
    #         validation_data_path='vote_multilabel_validation_'+str(i)+'.pkl',
    #         max_epochs=15,
    #         # reload_model = '../model_good/lstm_model_522_alltraining.npz',
    #         validFreq=100,
    #         saveFreq=100,
    #         saveto='/home/ubuntu/lstm/model_good/vote_multilabel_oversample_NEW' + str(i),
    #         patience=10
    #         # test_size=500,
    #     )

    #




    # web_path = 'vote_soft_multilabel_feature_all.npy'
    #
    # for i in range(5):
    #     # # web_path = 'vote_multilabel_feature_all.npy'
    #     web_path = 'vote_soft_multilabel_feature_all.npy'
    #     train_lstm_softtarget(
    #         # training_data_path='vote_softall_multilabel_training_'+str(i)+'_oversample_20_0.5.pkl',
    #         training_data_path='vote_softall_multilabel_training_'+str(i)+'.pkl',
    #         test_data_path='vote_softall_multilabel_test.pkl',
    #         validation_data_path='vote_softall_multilabel_validation_'+str(i)+'.pkl',
    #         max_epochs=15,
    #         # reload_model = '../model_good/lstm_model_522_alltraining.npz',
    #         validFreq=100,
    #         saveFreq=100,
    #         saveto='/home/ubuntu/lstm/model_good/vote_softall_multilabel_' + str(i),
    #         # saveto='/home/ubuntu/lstm/model_good/vote_softall_multilabel_oversample_20_0.5_' + str(i),
    #         patience=10
    #         # test_size=500,
    #     )




 #
 #    web_path = 'vote_multilabel_feature_all.npy'
 #    for file_name in [
 #        'vote_multilabel_0_alltraining'
 #        ,'vote_multilabel_1_alltraining',
 #        'vote_multilabel_2_alltraining','vote_multilabel_3_alltraining',
 #        # 'vote_multilabel_NEW_0_alltraining'
 # ]:
 #        test_prediction(file_name + '.npz', '_'.join(file_name.split('_')[:-1]),'vote_multilabel_test_all.pkl')
 #        lstm_recognition('_'.join(file_name.split('_')[:-1]), 'vote_multilabel_test_all.pkl')
    # file_name = 'vote_softall_multilabel'
    # lstm_recognition_crossvalidation(file_name, 'vote_multilabel_test_all.pkl')


    # web_path = 'vote_multilabel_feature_all.npy'
    # for file_name in [
    #     # 'vote_multilabel_oversample_20_0.5_0_alltraining','vote_multilabel_oversample_20_0.5_1_alltraining',
    #     # 'vote_multilabel_oversample_20_0.5_2_alltraining','vote_multilabel_oversample_20_0.5_3_alltraining',
    #     # 'vote_multilabel_oversample_20_0.5_4_alltraining'
    #     'vote_multilabel_0_alltraining','vote_multilabel_1_alltraining',
    #     'vote_multilabel_2_alltraining','vote_multilabel_3_alltraining',
    #     'vote_multilabel_4_alltraining'
    # ]:
    #     test_prediction(file_name + '.npz', '_'.join(file_name.split('_')[:-1]),'vote_multilabel_test_all.pkl')
    #     # lstm_recognition('_'.join(file_name.split('_')[:-1]), 'vote_multilabel_test_all.pkl')
    # # file_name = 'vote_multilabel_oversample_20_0.5'
    # # lstm_recognition_crossvalidation(file_name, 'vote_multilabel_test_all.pkl')
    #






    # web_path = 'vote_multilabel_feature_all.npy'
    # web_path = 'pec_feature_all_pca.npy'
    # for file_name in [
    #     # 'vote_multilabel_0_alltraining','vote_multilabel_1_alltraining',
    #     # 'vote_multilabel_2_alltraining','vote_multilabel_3_alltraining',
    #     # 'vote_multilabel_4_alltraining'
    #     'vote_multilabel_oversample_20_0.5_0_alltraining','vote_multilabel_oversample_20_0.5_1_alltraining',
    #     'vote_multilabel_oversample_20_0.5_2_alltraining','vote_multilabel_oversample_20_0.5_3_alltraining',
    #     'vote_multilabel_oversample_20_0.5_4_alltraining'
    # ]:
    #     test_prediction(file_name + '.npz', 'pec_'+ '_'.join(file_name.split('_')[:-1]), 'pec_vote_multilabel_all.pkl')
    #     lstm_recognition('pec_'+ '_'.join(file_name.split('_')[:-1]), 'pec_vote_multilabel_all.pkl')
    # file_name = 'pec_vote_multilabel_oversample_20_0.5'
    # lstm_recognition_crossvalidation(file_name, 'pec_vote_multilabel_all.pkl')

    #
    # web_path = 'pec_soft_feature_all_pca.npy'
    # for file_name in [
    #     # 'vote_softall_multilabel_0_alltraining','vote_softall_multilabel_1_alltraining',
    #     # 'vote_softall_multilabel_2_alltraining','vote_softall_multilabel_3_alltraining',
    #     # 'vote_softall_multilabel_4_alltraining'
    #     'vote_softall_multilabel_oversample_20_0.5_0_alltraining','vote_softall_multilabel_oversample_20_0.5_1_alltraining',
    #     'vote_softall_multilabel_oversample_20_0.5_2_alltraining','vote_softall_multilabel_oversample_20_0.5_3_alltraining',
    #     'vote_softall_multilabel_oversample_20_0.5_4_alltraining'
    # ]:
    #     test_prediction(file_name + '.npz', 'pec_'+ '_'.join(file_name.split('_')[:-1]), 'pec_vote_multilabel_soft_all.pkl')
    #     lstm_recognition('pec_new_'+ '_'.join(file_name.split('_')[:-1]), 'pec_vote_multilabel_soft_all.pkl')
    # file_name = 'pec_new_vote_multilabel_soft_oversample'
    # lstm_recognition_crossvalidation(file_name, 'pec_vote_multilabel_soft_all.pkl')

    #
    # web_path = 'pec_soft_feature_all_pca.npy'
    # # web_path = 'vote_multilabel_feature_all.npy'
    # for file_name in [
    #     'vote_softall_multilabel_0_alltraining','vote_softall_multilabel_1_alltraining',
    #     'vote_softall_multilabel_2_alltraining','vote_softall_multilabel_3_alltraining',
    #     'vote_softall_multilabel_4_alltraining'
    #     # 'vote_softall_multilabel_0_alltraining','vote_softall_multilabel_1_alltraining',
    #     # 'vote_softall_multilabel_2_alltraining','vote_softall_multilabel_3_alltraining',
    #     # 'vote_softall_multilabel_4_alltraining'
    # ]:
    #     test_prediction(file_name + '.npz', 'pec_'+ '_'.join(file_name.split('_')[:-1]), 'pec_vote_multilabel_soft_all.pkl')
    #     lstm_recognition('pec_'+ '_'.join(file_name.split('_')[:-1]), 'pec_vote_multilabel_soft_all.pkl')
    # file_name = 'pec_vote_softall_multilabel'
    # lstm_recognition_crossvalidation(file_name, 'pec_vote_multilabel_soft_all.pkl')

