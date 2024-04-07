# adapted from https://github.com/NVIDIA/tacotron2/blob/master/model.py
# add some modifications

from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
import numpy as np

from speaker_encoder import SpeakerEncoder
#import sys
#sys.path.append('/workspace')
#from TTS.speaker_encoder.model import SpeakerEncoder
#from TTS.utils.generic_utils import load_config

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        #print("[DECODER] encoder_embedding_dim-1:{}".format(self.encoder_embedding_dim))
        if hparams.with_language_embedding:
            self.encoder_embedding_dim += hparams.language_embedding_dim
        #print("[DECODER] encoder_embedding_dim-2:{}".format(self.encoder_embedding_dim))

        self.speaker_embedding_type = 0
        if hparams.with_speaker_embedding:
            self.encoder_embedding_dim += hparams.speaker_embedding_dim
            self.speaker_embedding_type = hparams.speaker_embedding_type
        #print("[DECODER] encoder_embedding_dim-3:{}".format(self.encoder_embedding_dim))
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet_input_dim = hparams.n_mel_channels * hparams.n_frames_per_step
        if self.speaker_embedding_type==1:
            self.prenet_input_dim += hparams.speaker_embedding_dim
        self.prenet = Prenet(
            self.prenet_input_dim,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + self.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, self.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths, speaker_embeds):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        if self.speaker_embedding_type==1:
            #print("[Decoder-Forward] decoder input:", decoder_inputs.size())
            #print("[Decoder-Forward] speaker embed:", speaker_embeds.size())
            speaker_embeds = speaker_embeds.repeat(decoder_inputs.size(0), 1, 1)
            #print("[Decoder-Forward] speaker embed 2:", speaker_embeds.size())
            decoder_inputs = torch.cat((decoder_inputs, speaker_embeds), dim=2)
            #print("[Decoder-Forward] decoder input cat:", decoder_inputs.size())
        decoder_inputs = self.prenet(decoder_inputs)
        # print("[Decoder-Forward] decoder input prenet:", decoder_inputs.size())

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, speaker_embed):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)
        if self.speaker_embedding_type==1:
            decoder_input = torch.cat((decoder_input, speaker_embed), dim=1)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output
            if self.speaker_embedding_type==1:
                decoder_input = torch.cat((decoder_input, speaker_embed), dim=1)

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2_ZS(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2_ZS, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

        # add language embedding
        self.with_language_embedding = hparams.with_language_embedding
        if self.with_language_embedding:
            self.language_embedding = nn.Embedding(
                hparams.n_languages, hparams.language_embedding_dim)

        # add speaker embedding
        self.with_speaker_embedding = hparams.with_speaker_embedding
        self.use_speaker_encoder = hparams.use_speaker_encoder
        if self.with_speaker_embedding:
            # speaker encoder
            #self.SE_config = load_config(hparams.SE_config_path)
            #self.speaker_encoder = SpeakerEncoder(**self.SE_config.model).cuda()
            self.speaker_encoder = SpeakerEncoder(
                hparams.se_input_dim, hparams.se_proj_dim, hparams.se_lstm_dim, hparams.se_num_lstm_layers).cuda()
            self.speaker_encoder.load_state_dict(torch.load(hparams.SE_model_path)['model'])
            for param in self.speaker_encoder.parameters():
                param.requires_grad = False

            if self.use_speaker_encoder == 0: # simple neural embedding lookup
                self.speaker_embedding = nn.Embedding(
                    hparams.n_speakers, hparams.speaker_embedding_dim)
                if hparams.speaker_embedding_file != "":  # use pre-defined lookup table
                    # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
                    pretrained_weight = np.load(hparams.speaker_embedding_file)
                    self.speaker_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
                    self.speaker_embedding.weight.requires_grad = False
                    # print("\n[Tacotron2_mlms] speaker_embedding weigt:", pretrained_weight)
                    # print("\n[Tacotron2_mlms] speaker_embedding ori:", self.speaker_embedding.weight)

    def get_speaker_embed(self, speaker_id):
        embed_speaker = self.speaker_embedding(speaker_id)
        return embed_speaker

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, language_ids, speaker_ids, speaker_embedds = batch

        #print("[TACOTRON2-EXT.parse_batch]")
        #print("mel_target", mel_padded.size())
        #print("gate_target", gate_padded.size())
        #print("text", text_padded.size())
        #print("input", input_lengths)
        #print("language_ids", language_ids.size())
        #print("speaker_ids", speaker_ids.size())
        #print("speaker_embedds_ori", speaker_embedds.size())

        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        language_ids = to_gpu(language_ids.data).long()
        speaker_ids = to_gpu(speaker_ids.data).long()
        speaker_embedds = to_gpu(speaker_embedds).float()

        if self.use_speaker_encoder == 0:  # use simple neural speaker embedding
            speaker_embedds = self.speaker_embedding(speaker_ids)
        #print("speaker_embed_target", speaker_embedds.size())

        if self.use_speaker_encoder == 2:  # use d-vector speaker encoder
            speaker_embedds = self.speaker_encoder.compute_embedding(mel_padded.permute(0,2,1))
        #print("speaker_embed_target", speaker_embedds.size())

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths,
             language_ids, speaker_ids, speaker_embedds),
            (mel_padded, gate_padded, speaker_embedds))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

            #print("PARSE OUTPUT")
            #print("Mel:", outputs[0].size())
            #print("Mel-postnet:", outputs[1].size())
            #print("Gate:", outputs[2].size())
            #print("Alingment:", outputs[3].size())
            #print("Speaker_embed:", outputs[4].size())

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths, \
            language_ids, speaker_ids, speaker_embedds = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        embedded_text = encoder_outputs

        if self.with_language_embedding:
            embedded_languages = self.language_embedding(language_ids)[:, None]
            embedded_languages = embedded_languages.repeat(1, embedded_text.size(1), 1)
            #print("[TACOTRON2-EXT.forward]")
            #print("text_lengths:", text_lengths)
            #print("text_inputs:", text_inputs.size())
            #print("embedded_inputs:", embedded_inputs.size())
            #print("encoder_output:", encoder_outputs.size())
            #print("embedded_language:", embedded_languages.size())
            #print("speaker_ids:",speaker_ids)
            #print("language_ids:", language_ids)
            #print(embedded_languages[0][0])
            #print(embedded_languages[1][0])
            encoder_outputs = torch.cat(
                (encoder_outputs, embedded_languages), dim=2)

        if self.with_speaker_embedding:
            embedded_speakers = speaker_embedds[:,None]
            #embedded_speakers_ori = self.speaker_embedding(speaker_ids)[:, None]
            embedded_speakers = embedded_speakers.repeat(1, embedded_text.size(1), 1)
            encoder_outputs = torch.cat(
                (encoder_outputs, embedded_speakers), dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths, speaker_embeds=speaker_embedds)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        speaker_embed_outputs =  self.speaker_encoder.compute_embedding(mel_outputs_postnet.permute(0,2,1))

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments,speaker_embed_outputs],
            output_lengths)

    def inference(self, inputs):
        #print("[Tacotron2-ext.inference] inputs:", inputs)
        #text, language_ids, speaker_ids, speaker_embedds = inputs

        text = inputs[0]
        #print("[Tacotron2-ext.inference] text:", text)
        if self.with_language_embedding:
            language_ids = inputs[1]
            #print("[Tacotron2-ext.inference] language_ids:", language_ids)
        if self.with_speaker_embedding:
            speaker_ids = inputs[2]
            #print("[Tacotron2-ext.inference] speaker_ids:", speaker_ids)
        #if self.use_speaker_encoder:
            speaker_embedds = inputs[3]
            #print("[Tacotron2-ext.inference] speaker_embedd:", speaker_embedd)

        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        embedded_text = encoder_outputs

        if self.with_language_embedding:
            embedded_languages = self.language_embedding(language_ids)[:, None]
            #print("Use Language Embedding\n",embedded_languages)
            embedded_languages = embedded_languages.repeat(1, embedded_text.size(1), 1)
            encoder_outputs = torch.cat(
                (encoder_outputs, embedded_languages), dim=2)

        if self.with_speaker_embedding:
            if speaker_embedds == None:
            #if self.use_speaker_encoder == 0: # use embedding lookup table
                # print("NOT Use Speaker Encoder")
                speaker_embedds = self.speaker_embedding(speaker_ids)
            #else: #  use external embedding
                #print("Use Speaker Encoder")

            embedded_speakers = speaker_embedds[:, None]
            #print(speaker_embedds[:, None])
            #print(self.speaker_embedding(speaker_ids)[:, None])
            embedded_speakers = embedded_speakers.repeat(1, embedded_text.size(1), 1)
            #print("Embed_size:{} Output_size:{}".format(embedded_speakers.size(),encoder_outputs.size()))
            encoder_outputs = torch.cat(
                (encoder_outputs, embedded_speakers), dim=2)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, speaker_embedds)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        speaker_embed_outputs =  self.speaker_encoder.compute_embedding(mel_outputs_postnet.permute(0,2,1))

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, speaker_embed_outputs])

        return outputs
