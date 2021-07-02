import torch
import torch.nn as nn

from neural_tdoa.models.dccrn.show import show_params, show_model
from neural_tdoa.models.dccrn.conv_stft import MultichannelConvSTFT
from neural_tdoa.models.dccrn.complexnn import (
    ComplexConv2d, NaiveComplexLSTM, ComplexBatchNorm
)


class TdoaDCCRN(nn.Module):
    def __init__(
        self,
        rnn_layers=2,
        rnn_units=128,
        win_len=400,
        win_inc=100,
        fft_len=512,
        win_type='hanning',
        masking_mode='E',
        use_clstm=False,
        use_cbn=False,
        kernel_size=5,
        kernel_num=[16, 32, 64, 128, 256, 256],
        num_channels=2,
        bidirectional=False
    ):
        """ 
            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        """

        super().__init__()

        self.fft_len = fft_len

        self.rnn_units = rnn_units
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size

        input_dim = num_channels*2 # 2 because of real and imag channels
        self.kernel_num = [input_dim]+kernel_num
        self.masking_mode = masking_mode
        self.use_clstm = use_clstm
        
        self.stft = MultichannelConvSTFT(win_len, win_inc, fft_len, win_type, 'complex')
        # self.stft = ConvSTFT(win_len, win_inc, fft_len, win_type, 'complex')

        self._init_convs(kernel_size, use_cbn)
        self._init_lstms(rnn_layers, bidirectional)

        show_model(self)
        show_params(self)
        self.flatten_parameters()

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, inputs):
        specs = self.stft(inputs)
        # real = specs[:, :, :self.fft_len//2+1]
        # imag = specs[:, :, self.fft_len//2+1:]
        # cspecs = torch.stack([real, imag], 1)
        #x = cspecs[:, :, 1:]
        x = specs[:, :, 1:]

        for _, layer in enumerate(self.convs):
            x = layer(x)

        
        x = self._forward_lstm(x)

        # Flatten in all dims except the batch
        x = x.flatten(start_dim=1)
        x = x.mean(1, keepdim=True)
        x = torch.sigmoid(x)
        # This discards all the complex structure, so maybe not the best idea.

        return x

    def _forward_lstm(self, x):
        batch_size, channels, dims, lengths = x.size()
        x = x.permute(3, 0, 1, 2)
        if self.use_clstm:
            r_rnn_in = x[:, :, :channels//2]
            i_rnn_in = x[:, :, channels//2:]
            r_rnn_in = torch.reshape(
                r_rnn_in, [lengths, batch_size, channels//2*dims])
            i_rnn_in = torch.reshape(
                i_rnn_in, [lengths, batch_size, channels//2*dims])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(
                r_rnn_in, [lengths, batch_size, channels//2, dims])
            i_rnn_in = torch.reshape(
                i_rnn_in, [lengths, batch_size, channels//2, dims])
            x = torch.cat([r_rnn_in, i_rnn_in], 2)
        else:
            # to [L, B, C, D]
            x = torch.reshape(x, [lengths, batch_size, channels*dims])
            x, _ = self.enhance(x)
            x = self.tranform(x)
            x = torch.reshape(x, [lengths, batch_size, channels, dims])

        x = x.permute(1, 2, 3, 0)
        
        return x

    def _init_convs(self, kernel_size, use_cbn):
        self.convs = nn.ModuleList()

        for idx in range(len(self.kernel_num) - 1):
            self.convs.append(
                _conv_block(idx, self.kernel_num, kernel_size, use_cbn)
            )

    def _init_lstms(self, rnn_layers, bidirectional):
        fac = 2 if bidirectional else 1
        hidden_dim = self.fft_len//(2**(len(self.kernel_num)))

        if self.use_clstm:
            rnns = []
            for idx in range(rnn_layers):
                rnns.append(
                    NaiveComplexLSTM(
                        input_size=hidden_dim *
                        self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=hidden_dim *
                        self.kernel_num[-1] if idx == rnn_layers-1 else None,
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim*self.kernel_num[-1],
                hidden_size=self.rnn_units,
                num_layers=2,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False
            )
            self.tranform = nn.Linear(
                self.rnn_units * fac, hidden_dim*self.kernel_num[-1])




def _conv_block(idx, kernel_num, kernel_size, use_cbn):
    block = nn.Sequential(
        #nn.ConstantPad2d([0, 0, 0, 0], 0),
        ComplexConv2d(
            kernel_num[idx],
            kernel_num[idx+1],
            kernel_size=(kernel_size, 2),
            stride=(2, 1),
            padding=(2, 1)
        ),
        nn.BatchNorm2d(
            kernel_num[idx+1]) if not use_cbn else ComplexBatchNorm(kernel_num[idx+1]),
        nn.PReLU()
    )

    return block


if __name__ == '__main__':
    torch.manual_seed(10)
    torch.autograd.set_detect_anomaly(True)
    inputs = torch.randn([10, 4, 16000*4]).clamp_(-1, 1)

    net = TdoaDCCRN(rnn_units=256, masking_mode='E', use_clstm=True,
                kernel_num=[32, 64, 128, 256, 256, 256], num_channels=4)
    out = net(inputs)
    print(out.shape)
    #loss = net.loss(out_wav, labels, loss_mode='SI-SNR')
    #print(loss)
