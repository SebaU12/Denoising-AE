import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, input_channels, num_layers, stride_interval, cincrease_interval, base_channel, debug) -> None:
        super(Encoder, self).__init__()
        self.debug = debug
        self.num_layers = num_layers
        self.stride_interval = stride_interval
        in_channels = input_channels

        self.conv_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        
        out_channels = base_channel
        remain = 0
        p_dropout = 0.20
        for i in range(num_layers):
            if i % cincrease_interval == 0 and i != 0:
                out_channels = out_channels * 2
                remain = 0

            stride = 2 if (i + 1) % stride_interval == 0 else 1
            
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.conv_layers.append(conv_layer)
            
            activation_layer = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15)
            )
            self.activation_layers.append(activation_layer)

            if (stride == 2 and p_dropout != 0.40):
                p_dropout += 0.10
                p_dropout = round(p_dropout, 2)
            
            in_channels = out_channels
            remain += 1
        self.remain = remain
    
    def forward(self, x):
        if self.debug==True:
            print("-----------------------------------------")
            print(f'Ecoder Input: {x.size()}')
        cont = 0
        skip_conn_list = []
        skip_conn_list.append(x)
        if self.debug==True:
            print(f'Saving skip layer {cont}')
        for conv_layer, activation_layer in zip(self.conv_layers, self.activation_layers):
            x = conv_layer(x)
            x = activation_layer(x)
            if self.debug==True:
                print(f'Layer {cont+1} output: {x.size()}')
            if (cont+1) % self.stride_interval == 0 and (cont+1) != self.num_layers:
                if self.debug==True:
                    print(f'Saving skip layer {cont+1}')
                skip_conn_list.append(x)
            cont += 1
        if self.debug==True:
            print("-----------------------------------------")
        return x, skip_conn_list
    
class Decoder(nn.Module):
    def __init__(self, output_channels, num_layers, stride_interval, cdecrease_interval, base_channel, remain, debug) -> None:
        super(Decoder, self).__init__()
        self.debug = debug
        self.num_layers = num_layers
        self.stride_interval = stride_interval
        
        in_channels = base_channel
        out_channels = base_channel 
        self.conv_transpose_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()

        cont_cinterval = remain-1
        cont_sinterval = 0
        p_dropout = 0.50
        for i in range(num_layers-1):
            if cont_cinterval == 0:
                out_channels = out_channels//2
                cont_cinterval = cdecrease_interval 
            cont_cinterval -= 1
            if cont_sinterval == 0:
                stride = 2 
                cont_sinterval = stride_interval
            else: 
                stride = 1
            cont_sinterval -= 1

            if (stride == 2 and p_dropout != 0.20):
                p_dropout -= 0.10
                p_dropout = round(p_dropout, 2)
            output_padding = (stride - 1) if stride == 2 else 0

            conv_transpose_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding)
            self.conv_transpose_layers.append(conv_transpose_layer)

            activation_layer = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )

            self.activation_layers.append(activation_layer)

            in_channels = out_channels
        
        self.final_conv_transpose = nn.ConvTranspose2d(in_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.final_norm = nn.Sequential(
            nn.BatchNorm2d(output_channels),
            nn.Sigmoid()  
        )
    
    def forward(self, x, skip_conns):
        if self.debug==True:
            print("-----------------------------------------")
            print(f'Decoder Input: {x.size()}')
        cont = 0
        total_skips = len(skip_conns)
        if self.debug==True:
            for i in range(total_skips):
                print(f'Skip connection {i+1}: {skip_conns[i].size()}')
        for conv_transpose_layer, activation_layer in zip(self.conv_transpose_layers, self.activation_layers):
            x = conv_transpose_layer(x)
            if (cont+1) % self.stride_interval == 0 and cont != 0:
                if self.debug == True:
                    print(f'Add skip conn Layer {self.num_layers-cont}: {skip_conns[total_skips-1].size()}')
                x += skip_conns[total_skips-1]
                total_skips -= 1
            x = activation_layer(x)
            if self.debug==True:
                print(f'Layer {self.num_layers-cont} output: {x.size()}')
            cont += 1
        if self.debug == True:
            print(f'Add skip conn Layer {self.num_layers-cont}: {skip_conns[0].size()}')
        x = self.final_conv_transpose(x) + skip_conns[0]
        x = self.final_norm(x)
        if self.debug==True:
            print(f'Layer {self.num_layers-cont} output: {x.size()}')
            print("-----------------------------------------")
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_channels, num_layers, stride_interval, channels_interval, base_channel, debug) -> None:
        super(Autoencoder, self).__init__()
        self.debug = debug
        base = base_channel
        self.encoder = Encoder(input_channels, num_layers, stride_interval, channels_interval, base, debug)
        if (num_layers%channels_interval== 0):
            exp = (num_layers//channels_interval)-1 
        else:
            exp = (num_layers//channels_interval)
        for _ in range(exp):
            base = base * 2
        remain_channel = self.encoder.remain
        self.decoder = Decoder(input_channels, num_layers, stride_interval, channels_interval, base, remain_channel, debug)

    def forward(self, x):
        x, skip_conns = self.encoder(x)
        x = self.decoder(x, skip_conns)
        return x