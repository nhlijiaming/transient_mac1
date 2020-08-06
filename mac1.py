import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import mac1_model

class Machine1:
    def __init__(self, n_layers=2, hidden_dim=[400,400],n_training_sample=1600, data_len=900, batchsize=400, gradient_clip=0.0001, teacher_forcing_ratio_decay=True, teacher_forcing_ratio=0.75, data_interpolation_rate=None, verbose_output=False):

        self.n_training_sample = n_training_sample
        self.data_len = data_len
        self.batchsize = batchsize
        
        self.gradient_clip = gradient_clip # options: 0.002, 0.0005, 0.0001

        def MSELoss(s1, s2, weight=0.0):
            return (F.mse_loss(s1, s2))# + weight*F.mse_loss(s1[:,:,1], s2[:,:,1]))
#         self.criterion = nn.MSELoss()
#         print('Weighted * 10')
        self.criterion = MSELoss
        self.trained_epochs = 1
        self.loss_curve = []
        self.teacher_forcing_ratio_decay = True
        self.teacher_forcing_ratio = 0.75
        
        # Define constants
        self.data_format = self.data_structure()
        self.label_format = self.label_structure()
        
        # Load data
        self.load_data(self.n_training_sample, apply_mac_ang_filter=True, data_interpolation_rate=data_interpolation_rate, verbose_output=verbose_output)
        
        # Setting up random seed
        torch.manual_seed(20200410)
        if torch.cuda.is_available():
            print('GPU is available. Deploying GPU.')
            self.use_GPU = True
            self.device = torch.device('cuda')

            torch.cuda.empty_cache()
            torch.cuda.manual_seed(20200410)
            torch.cuda.manual_seed_all(20200410)
            torch.backends.cudnn.deterministic = True
        else:
            print('GPU is NOT available. CPU will be deployed.')
            self.use_GPU = False
            self.device = torch.device('cpu')

        # Construct and initialize the neural network model
        self.model = mac1_model.LSTMNetwork(n_layers=n_layers, hidden_dim=hidden_dim, device=self.device).float()
        self.model.apply(self.weights_init)
        self.model = self.model.to(self.device)

        # Set up optimizer
        # optimizer = optim.SGD(model.parameters(), lr=0.01*2, momentum=0.9, nesterov=True)
        self.optimizer = optim.Adam(self.model.parameters())
        # optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

        # Transfer data to pyTorch
        self.train_data_torch = torch.from_numpy(self.train_data).float()
        self.train_label_torch = torch.from_numpy(self.train_label).float()
        self.test_data_torch = torch.from_numpy(self.test_data).float()
        self.test_label_torch = torch.from_numpy(self.test_label).float()

        
    def load_pretrained(self, file):
        # Load saved file
        checkpoint = torch.load(file, map_location=self.device)
        
        # Load trained parameters 
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Parameters loaded in.')

        # Test
        rse_train = self.evaluation(self.train_data_torch, self.train_label_raw)
        rse_test = self.evaluation(self.test_data_torch, self.test_label_raw)
        print('Before training: train eval RSE = %.3e, test eval RSE = %.3e' % (np.median(rse_train,axis=0).mean(), np.median(rse_test,axis=0).mean()))

        # to compatible with random seed
        for i in range(self.trained_epochs, self.trained_epochs+checkpoint['epoch']):
            if self.teacher_forcing_ratio_decay:
                self.teacher_forcing_ratio *= 0.9975 #  0.9966
            for j in range(0, self.n_training_sample, self.batchsize):
                use_teacher_forcing = np.random.random()<self.teacher_forcing_ratio
        self.trained_epochs = self.trained_epochs+checkpoint['epoch']
        self.loss_curve = checkpoint['loss_curve']
        

    def load_data(self, n_training_sample, apply_mac_ang_filter, data_interpolation_rate, verbose_output):
        # Data pre-processing
        np.random.seed(20200412)
        data_len = self.data_len

        sample_idx = np.random.permutation(5277)
        self.sample_idx = sample_idx
        
        # Loading .mat file
        import scipy.io as sio
        self.data = sio.loadmat('data_mac1_full_wBus2.mat')['data']
        data = self.data
        # format: data['variable name'] [0] [sample index] [0] [time step]
        # e.g.: print(data['bus_v']     [0]     [432]      [0]   [124])
        
        if verbose_output:
            print('Components in the dataset:', data.dtype.names)
            print('Dataset has {:d} trajectories.'.format(data.shape[1]))
        
        # Select training data
        train_data = np.zeros((data_len, n_training_sample, 11), dtype=np.float64)
        train_label_raw = np.zeros((data_len, n_training_sample, 7), dtype=np.float64)
        n_entry = 0
        for i in sample_idx[:4500]:
            filename = data['filename'][0][i][0]
            if apply_mac_ang_filter and (filename.find('a5a') > -1 or filename.find('a5.') > -1):
                if verbose_output:
                    print('{:s} was skipped because it contains a line-5 fault.'.format(data['filename'][0][i][0]))
                continue
            
            tmp_train_data, tmp_train_label = self.extract_data(i, data_len, data_interpolation_rate)

            train_data[:, n_entry, :] = tmp_train_data
            train_label_raw[:, n_entry, :] = tmp_train_label
            n_entry += 1
            if n_entry >= n_training_sample:
                break
        train_data = train_data[:, :n_entry, :]
        train_label_raw = train_label_raw[:, :n_entry, :]
        del tmp_train_data
        del tmp_train_label

        # Select validation data
        test_data = np.zeros((data_len, 400, 11), dtype=np.float64)
        test_label_raw = np.zeros((data_len, 400, 7), dtype=np.float64)

        n_entry = 0
        for i in sample_idx[4800:]:
            filename = data['filename'][0][i][0]
            if apply_mac_ang_filter and (filename.find('a5a') > -1 or filename.find('a5.') > -1):
                if verbose_output:
                    print('{:s} was skipped because it contains a line-5 fault.'.format(data['filename'][0][i][0]))
                continue

            tmp_test_data, tmp_test_label = self.extract_data(i, data_len, data_interpolation_rate)
                
            test_data[:, n_entry, :] = tmp_test_data
            test_label_raw[:, n_entry, :] = tmp_test_label
            n_entry += 1
            if n_entry >= 400:
                break

        test_data = test_data[:, :n_entry, :]
        test_label_raw = test_label_raw[:, :n_entry, :]
        del tmp_test_data
        del tmp_test_label

        # normalize data
        data_mean = np.mean(train_data.reshape(n_training_sample*data_len,-1), axis=0)
        data_std = np.std(train_data.reshape(n_training_sample*data_len,-1), axis=0)
        data_std[np.less(data_std, 1e-7)] = 1e-7
        
#         data_mean = np.min(train_data.reshape(n_training_sample*data_len,-1), axis=0)
#         data_std = np.max(train_data.reshape(n_training_sample*data_len,-1), axis=0) - data_mean
#         data_std[np.less(data_std, 1e-7)] = 1e-7
        
        train_data = (train_data - data_mean) / data_std
        test_data = (test_data - data_mean) / data_std

        # normalize label
        label_mean = data_mean[[0,1,4,5,6,7,8]]
        label_std = data_std[[0,1,4,5,6,7,8]]
        train_label = (train_label_raw - label_mean) / label_std
        test_label = (test_label_raw - label_mean) / label_std
        
        if verbose_output:
            print('Train data set size:', train_data.shape)
            print('Train label set size:', train_label.shape)
            print('Train data set size:', train_data.shape)
            print('Test label set size:', test_label.shape)

        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label
        self.train_label_raw = train_label_raw
        self.test_label_raw = test_label_raw
        self.label_mean = label_mean
        self.label_std = label_std
        self.data_mean = data_mean
        self.data_std = data_std
        
    def extract_data(self, index, data_len, data_interpolation_rate, clip=None):
        # Extract quantities from index-th case
        data = self.data
        
        bus_v = data['bus_v'][0][index].reshape(-1, 1)
        cur = data['cur'][0][index].reshape(-1, 1)
#         bus_freq = data['bus_freq'][0][index].reshape(-1, 1)
        mac_ang = data['mac_ang'][0][index].reshape(-1, 1)
        mac_spd = data['mac_spd'][0][index].reshape(-1, 1)
        pelect = data['pelect'][0][index].reshape(-1, 1)
        pmech = data['pmech'][0][index].reshape(-1, 1)
        qelect = data['qelect'][0][index].reshape(-1, 1)
        bus2 = data['bus2'][0][index].reshape(-1, 1)
        length = qelect.shape[0]

        bus_v_ang = np.unwrap(np.angle(bus_v).reshape(-1)).reshape(-1,1)
        cur_ang = np.unwrap(np.angle(cur).reshape(-1)).reshape(-1,1)
        bus2_ang = np.unwrap(np.angle(bus2).reshape(-1)).reshape(-1,1)
        tmp_data = np.hstack([np.abs(bus_v), bus_v_ang, np.abs(cur), cur_ang, mac_ang, mac_spd, pelect, pmech, qelect, np.abs(bus2), bus2_ang])
        tmp_data = tmp_data[:data_len+1, :]
        
        # Interpolation
        if data_interpolation_rate:
            # Move the fault time position
            if clip is None:
                clip = np.random.randint(low=50,high=97)
            tmp_data = tmp_data[clip:, :]
            
            tmp_data_p = np.zeros((data_len+1, tmp_data.shape[1]))
            for i in range(tmp_data.shape[1]):
                x = np.arange(0.0,data_len/100,0.01/data_interpolation_rate)
                xp = np.arange(0.0,tmp_data[:, i].shape[0]/100-1e-4,0.01)
                tmp_data_p[:, i] = np.interp(x, xp, tmp_data[:, i])[:data_len+1]
            tmp_data = tmp_data_p

        # delete the first sample(shift the curve left)
        tmp_label = np.delete(tmp_data, 0, 0)  

        # delete the currents(we don't need to predict that)
        tmp_label = np.delete(tmp_label, np.arange(9, 11), 1)
        tmp_label = np.delete(tmp_label, np.arange(2, 4), 1)
        
        # delete the last sample because there's no corresponding label
        tmp_data = np.delete(tmp_data, -1, 0)  

        return tmp_data, tmp_label
        
    def test(self, data_torch, label_torch, test_batchsize = 200):
        model, device = self.model, self.device
        loss = None
        prediction = None
        
        self.model.eval()
        with torch.no_grad():
            for j in range(0, data_torch.shape[1], test_batchsize):
                # Construct mini-batch data
                inputs_torch = data_torch[:, j:j+test_batchsize, :]
                labels_torch = label_torch[:, j:j+test_batchsize, :]

                inputs_torch = inputs_torch.to(device)
                labels_torch = labels_torch.to(device)

                # Forward
                pred_torch, _ = self.model(inputs_torch)
                loss_torch = self.criterion(pred_torch, labels_torch)

                if self.use_GPU:
                    pred_torch = pred_torch.cpu()
                pred = pred_torch.data.numpy()

                loss = np.array([loss_torch.item()]) if loss is None else np.hstack([loss, loss_torch.item()])
                prediction = pred if prediction is None else np.concatenate([prediction, pred], axis=1)
        return loss.mean(), prediction

    def evaluation(self, eval_data_torch, eval_label_raw, eval_batchsize = 200):
        model, device = self.model, self.device
        t_max = self.data_len
        samples = eval_data_torch.shape[1]
        rse = np.zeros((samples,7))

        model.eval()
        with torch.no_grad():
            for i in range(0,samples,eval_batchsize):
                rnn_states = model.initial_states()
                output_data = np.zeros((t_max, eval_batchsize, 7))
                input_data_torch = eval_data_torch[0, i:i+eval_batchsize, :].reshape(1,eval_batchsize,-1).to(device)

                for t in range(0, t_max):
                    output_data_torch, rnn_states = model(input_data_torch, rnn_states)
                    output_data[t,:] = output_data_torch.data.cpu().numpy()
                    
                    # construct mini-batch data
                    if t<t_max-1:
                        input_data_torch = torch.cat((output_data_torch[0,:,0:2],
                                                  eval_data_torch[t+1,i:i+eval_batchsize,2:4].to(device), 
                                                  output_data_torch[0,:,2:7], 
                                                  eval_data_torch[t+1,i:i+eval_batchsize,9:11].to(device)
                                                 ), dim=1).reshape(1,eval_batchsize,-1)
                
                # de-normalize before computing Relative RMSE
                output_data = output_data * self.label_std + self.label_mean

                for b in range(eval_batchsize):
                    for q in range(7):
                        
                        # skip pmech, which is a constant
                        if q == 5:
                            continue
                        
                        truth_curve = eval_label_raw[:t_max, i+b, q]
                        pred_curve = output_data[:t_max, b, q]
                        rse[i+b,q] = np.linalg.norm(truth_curve - pred_curve, 2) / (np.linalg.norm(truth_curve-truth_curve.mean(), 2) + 1e-6)
        return rse

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            #         torch.nn.init.kaiming_normal_(m.weight)
            #         torch.nn.init.normal_(m.weight, mean=0, std=0.001)
            torch.nn.init.normal_(m.bias, mean=0, std=0.001)
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    #         nn.init.kaiming_uniform_(m.bias, mode='fan_in', nonlinearity='relu')

        if isinstance(m, nn.BatchNorm1d):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.ones_(m.weight)
            
    def train_for_one_epoch(self):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        criterion = self.criterion
        data_len = self.data_len
        batchsize = self.batchsize
        loss_curve = self.loss_curve
        
        train_data_torch = self.train_data_torch
        train_label_torch = self.train_label_torch

        model.train()
        if self.teacher_forcing_ratio_decay:
            self.teacher_forcing_ratio *= 0.9975 #  0.9966

        for j in range(0, self.n_training_sample, self.batchsize):
            loss = 0
            rnn_states = model.initial_states()

            use_teacher_forcing = np.random.random()<self.teacher_forcing_ratio
            if use_teacher_forcing:
                # with total teacher forcing, 20x faster
                inputs = train_data_torch[:data_len, j:j+batchsize, :].reshape(data_len,batchsize,-1).to(device)
                outputs, rnn_states = model(inputs, rnn_states)
                labels = train_label_torch[:, j:j+batchsize, :].to(device)
                loss = criterion(outputs, labels)
            else:
                for t in range(data_len):
                    # construct input data from previous output and external dataset
                    inputs = train_data_torch[t, j:j+batchsize, :].reshape(1,batchsize,-1).clone().to(device)
                    if t>0:
                        # teacher forcing: replacing inputs with ground truth
                        inputs[:,:,[0,1,4,5,6,7,8]] = outputs[0,:,[0,1,2,3,4,5,6]].detach()

                    # forward
                    outputs, rnn_states = model(inputs, rnn_states)
                    
                    # compute loss
                    labels = train_label_torch[t, j:j+batchsize, :].reshape(1,batchsize,-1).to(device)
                    loss += criterion(outputs, labels)
                loss /= data_len
                
            # backward
            optimizer.zero_grad()
            loss.backward()

            # apply gradient clip
            torch.nn.utils.clip_grad_value_(model.parameters(), self.gradient_clip)
            
            # update parameter
            optimizer.step()
            
        self.trained_epochs = self.trained_epochs + 1
        
    class data_structure:
        def __init__(self):
            self.bus_v_mag = 0
            self.bus_v_ang = 1
            self.cur_mag = 2
            self.cur_ang = 3
            self.mac_ang = 4
            self.mac_spd = 5
            self.pelect = 6
            self.pmech = 7
            self.qelect = 8
            self.bus2_mag = 9
            self.bus2_ang = 10
            self.bus_v = [self.bus_v_mag, self.bus_v_ang]
            self.cur = [self.cur_mag, self.cur_ang]
            self.bus2 = [self.bus2_mag, self.bus2_ang]

    class label_structure:
        def __init__(self):
            self.bus_v_mag = 0
            self.bus_v_ang = 1
            self.mac_ang = 2
            self.mac_spd = 3
            self.pelect = 4
            self.pmech = 5
            self.qelect = 6
            self.bus_v = [self.bus_v_mag, self.bus_v_ang]