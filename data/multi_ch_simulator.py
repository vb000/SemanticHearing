import pyroomacoustics as pra
import numpy as np
import random
import json
import os, glob
import sofa
import torch
import torchaudio.transforms as AT
from data.utils import read_audio_file

from scipy.signal import convolve
from scipy.ndimage import convolve1d


import time

class BaseSimulator(object):
    def __init__(self):    
        pass
    
    def preprocess(self, audio):
        return audio
        
    def postprocess(self, audio):
        return audio
            
    def randomize_sources(self, num_sources):
        pass
    
    def get_metadata(self):
        metadata = {}
        
        metadata['duration'] = self.D
        metadata['sofa'] = self.sofa
        
        metadata['mic_positions'] = self.mic_positions

        metadata['sources'] = []
        for i, source_id in enumerate(self.source_order):
            source = {'position':self.source_positions[i], 
                      'order':source_id,
                      'hrtf_index':self.hrtf_indices[i],
                      'label':self.source_labels[i]}
            metadata['sources'].append(source)

        metadata['num_background'] = self.num_background_sources
        
        return metadata

    def save(self, path):
        metadata = self.get_metadata()
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def simulate(self, audio: np.ndarray) -> np.ndarray:
        """
        Simulates RIR
        audio: (C x T)
        """
        num_sources = audio.shape[0]
       
        #t1 = time.time()

        rirs = self.get_rirs()

        #t2 = time.time()

        #t_rir = t2 - t1
        
        #t1 = time.time()
        x = self.preprocess(audio)

        output = []
        for i in range(num_sources):
            rir = rirs[i]
            waveform = x[i]
            
            left = convolve(waveform, rir[0])
            left = self.postprocess(left)
            
            right = convolve(waveform, rir[1])
            right = self.postprocess(right)
            
            binaural = np.stack([left, right])
            output.append(binaural)

        output = np.array(output, dtype=np.float32)
        #t2 = time.time()

        #t_convolve = t2 - t1

        #print('RIR time:', t_rir)
        #print('Convolution time:', t_convolve)

        return output
    
    def initialize_room_with_random_params(self,
                                           num_sources: int,
                                           duration: float,
                                           ann_list: list, 
                                           nbackground_sources: int = 1):
        self.D = duration

        self.source_labels = []
        for i in range(num_sources):
            self.source_labels.append(ann_list[i])

        # Randomize source choose order
        # First k sources correspond to background sources
        # Next n - k sources are foreground sources
        n = num_sources
        k = nbackground_sources
        self.source_order = [i for i in range(n - k)]
        np.random.shuffle(self.source_order)
        self.source_order = [i for i in range(n - k, n)] + self.source_order

        self.num_background_sources = k

        return self

    def seed(self, seed_value):
        np.random.seed(seed_value)
        random.seed(seed_value)
        
class CATTRIR_Simulator(BaseSimulator):
    def __init__(self, dset_text_file, **kwargs) -> None:
        super().__init__()
        
        dset_dir = os.path.dirname(dset_text_file)
        with open(dset_text_file, 'r') as f:
            self.rt60_list = f.read().split('\n')
            self.rt60_dirs = [os.path.join(dset_dir, x) for x in self.rt60_list]

    def randomize_sources(self, num_sources):
        source_positions = []
        hrtf_indices = []
        rirs = sorted(os.listdir(self.room_dir))
        random_source_rir_wavs = random.sample(rirs, num_sources)
        
        angles = []
        for f in random_source_rir_wavs:
            angle = int(f[f.rfind('_')+1:-4])
            angles.append(angle)
        
        for i in range(num_sources):
            pos = [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]
            source_positions.append(pos)
            hrtf_indices.append(angle)

        return source_positions, hrtf_indices
    
    def get_rirs(self):
        num_sources = len(self.source_positions)
        rt60 = os.path.basename(self.room_dir)
        
        rirs = []
        for i in range(num_sources):
            path = os.path.join(self.room_dir, f'CATT_{rt60}_{self.hrtf_indices[i]}.wav')
            rir = read_audio_file(path, 44100)
            rirs.append(rir.astype(np.float32))
        
        return rirs
            
    def initialize_room_with_random_params(self,
                                           num_sources: int,
                                           duration: float,
                                           ann_list: list, 
                                           nbackground_sources: int = 1):
        
        self.room_dir = self.rt60_dirs[np.random.randint(len(self.rt60_dirs))]
        self.sofa = self.room_dir # TODO: Implement this better
        
        self.mic_positions = [[0, 0.9, 0], [0, -0.9, 0]]
        self.source_positions, self.hrtf_indices = self.randomize_sources(num_sources) 
        
        return super().initialize_room_with_random_params(num_sources, 
                                                       duration,
                                                       ann_list,
                                                       nbackground_sources)
        
class SOFASimulator(BaseSimulator):
    def __init__(self, sofa_text_file, **kwargs) -> None:
        super().__init__()
        self.hrtf_cache = {}
        self.sofa_dict = {}
        sofa_dir = os.path.dirname(sofa_text_file)
        with open(sofa_text_file, 'r') as f:
            self.subject_sofa_list = f.read().split('\n')
            self.sofa_files = [os.path.join(sofa_dir, x) for x in self.subject_sofa_list]
            
            for f in self.sofa_files:
                self.sofa_dict[f] = sofa.Database.open(f)
        
        self.kwargs = kwargs

    def initialize_room_with_random_params(self,
                                           num_sources: int,
                                           duration: float,
                                           ann_list: list, 
                                           nbackground_sources: int = 1):
        
        self.sofa = self.sofa_files[np.random.randint(len(self.sofa_files))]
        self.HRTF = self.sofa_dict[self.sofa]#sofa.Database.open(self.sofa)
        mic_positions = self.HRTF.Receiver.Position.get_values(system="cartesian")[..., 0]
        self.mic_positions = mic_positions.tolist()
        self.source_positions, self.hrtf_indices = self.randomize_sources(num_sources) 
        
        return super().initialize_room_with_random_params(num_sources, 
                                                       duration,
                                                       ann_list,
                                                       nbackground_sources)
    def get_rirs(self):
        num_sources = len(self.source_positions)
        rirs = []
        for i in range(num_sources):
            key = self.sofa + str(sorted(list(self.hrtf_indices[i].items())))
            #print('KEY', key)
            if key in self.hrtf_cache:
                rir = self.hrtf_cache[key]
            else:
                rir = self.HRTF.Data.IR.get_values(indices=self.hrtf_indices[i]).astype(np.float32)
                self.hrtf_cache[key] = rir.copy()
            rirs.append(rir)
        return rirs
    
class CIPIC_Simulator(SOFASimulator):
    def randomize_sources(self, num_sources):
        source_positions = []
        hrtf_indices = []
        random_source_positions = random.sample(range(self.HRTF.Dimensions.M), num_sources)
        for i in range(num_sources):
            sofa_indices = {"M":random_source_positions[i]}
            pos = self.HRTF.Source.Position.get_values(system="cartesian", indices=sofa_indices).tolist()
            source_positions.append(pos)
            hrtf_indices.append(sofa_indices)

        return source_positions, hrtf_indices

    
class CIPIC_HRTF_Simulator(CIPIC_Simulator): pass

class BRIR48kHz_Simulator(CIPIC_HRTF_Simulator):
    def __init__(self, sofa_text_file, **kwargs):
        super().__init__(sofa_text_file, **kwargs)
        self.presampler = AT.Resample(self.kwargs['sr'], 48000)
        self.postsampler = AT.Resample(48000, self.kwargs['sr'])
    
    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        audio = self.presampler(torch.from_numpy(audio))
        return audio.numpy()
    
    def postprocess(self, audio: np.ndarray) -> np.ndarray:
        audio = self.postsampler(torch.from_numpy(audio))
        return audio.numpy()

    
# Salford-BBC Spatially-sampled Binaural Room Impulse Responses
# https://usir.salford.ac.uk/id/eprint/30868/
class SBSBRIR_Simulator(BRIR48kHz_Simulator):    
    def randomize_sources(self, num_sources):
        source_positions = []
        hrtf_indices = []
        
        random_source_positions = random.sample(range(self.HRTF.Dimensions.E), num_sources)
        random_measurement_rotation = np.random.randint(self.HRTF.Dimensions.M)
        for i in range(num_sources):
            #sofa_indices = {"M":0, "E":0}
            sofa_indices = {"M":random_measurement_rotation, "E":random_source_positions[i]}
            pos = self.HRTF.Emitter.Position.get_values(system="cartesian", indices=sofa_indices).tolist()
            source_positions.append(pos)
            hrtf_indices.append(sofa_indices)

        return source_positions, hrtf_indices
    
    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        audio = super().preprocess(audio)
        return audio * 15 # Gain because RIRs are very low for some reason
    
# Real Room BRIRs
# https://github.com/IoSR-Surrey/RealRoomBRIRs
class RRBRIR_Simulator(BRIR48kHz_Simulator): pass
    
class Multi_Ch_Simulator(BaseSimulator):
    # simulators = [CIPIC_Simulator]
    # simulators = [ CATTRIR_Simulator]
    # simulators = [ SBSBRIR_Simulator]
    # simulators = [RRBRIR_Simulator]
    # simulators = [SBSBRIR_Simulator, RRBRIR_Simulator, CATTRIR_Simulator] # UNCOMMENT FOR REVERBED HRTF ONLY
    def __init__(self, hrtf_dir, dset_type: str, sr: int, reverb: bool = True) -> None:
        self.hrtf_dir = hrtf_dir
        self.dset = dset_type
        self.sr = sr
        
        if reverb:
            simulators = [CIPIC_Simulator, SBSBRIR_Simulator, RRBRIR_Simulator, CATTRIR_Simulator] 
        else:
            simulators = [CIPIC_Simulator]
        
        #simulators = [SBSBRIR_Simulator]
        self.simulators = [sim(os.path.join(self.hrtf_dir, sim.__name__[:-len("_Simulator")], self.dset + '_hrtf.txt'), sr=self.sr) for sim in simulators]

    def get_random_simulator(self) -> BaseSimulator:
        sim = random.choice(self.simulators)
        #print("Using simulator", type(sim))
        return sim#(os.path.join(self.hrtf_dir, sim.__name__[:-len("_Simulator")], self.dset + '_hrtf.txt'),sr=self.sr)

class PRASimulator(object):
    def __init__(self,
                 n_mics = 2,
                 min_absorption=0.6,
                 max_absorption=1,
                 fs=44100,
                 max_order=15,
                 mean_mic_distance=13.9,
                 mic_distance_var=0.7,
                 mic_array_keepout=0.5,
                 min_room_length=6,
                 max_room_length=8,
                 min_room_width=6,
                 max_room_width=8) -> None:
        """
        Mic distance is by default the average of the
        median Bitragion Breadth for men and women
        """
        # Constant across samples
        self.M = n_mics
        self.fs = fs
        self.K = mic_array_keepout
        self.max_order = max_order
        self.min_absorption = min_absorption
        self.max_absorption = max_absorption
        self.R = mean_mic_distance
        self.V = mic_distance_var

        self.min_room_length = min_room_length
        self.max_room_length = max_room_length
        self.min_room_width = min_room_width
        self.max_room_width = max_room_width

    def initialize_room_with_random_params(self, num_sources: int, duration: float):
        self.D = duration
        self.mic_distance = np.random.normal(self.R, scale=self.V ** 0.5) * 1e-2
        self.mic_positions = [[-self.mic_distance/2, 0], [self.mic_distance/2, 0]]
        self.absorption = np.random.uniform(self.min_absorption, self.max_absorption)
        
        self.L = np.random.uniform(self.min_room_length, self.max_room_length)
        self.W = np.random.uniform(self.min_room_width, self.max_room_width)

        self.left_wall = -self.L / 2
        self.right_wall = self.L / 2
        
        self.bottom_wall = -self.W / 2
        self.top_wall = self.W / 2
        
        self.source_positions = []
        for i in range(num_sources):
            source_pos = self._get_random_source_pos(self.left_wall,
                                                    self.right_wall,
                                                    self.bottom_wall,
                                                    self.top_wall,
                                                    self.K)
            self.source_positions.append(source_pos)

        # Randomize source choose order
        self.source_order = [i for i in range(num_sources - 1)]
        np.random.shuffle(self.source_order)
        self.source_order = [num_sources-1] + self.source_order # Background is always last

        return self

    def intialize_from_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.D = metadata['duration']
        self.M = metadata['n_mics']
        self.fs = metadata['sampling_rate']
        self.max_order = metadata['max_order']
        self.absorption = metadata['absorption']
        self.mic_distance = metadata['mic_distance']
        self.mic_positions = metadata['mic_positions']

        room_desc = metadata['room']
        self.L = room_desc['length']
        self.W = room_desc['width']

        self.left_wall = -self.L / 2
        self.right_wall = self.L / 2
        
        self.bottom_wall = -self.W / 2
        self.top_wall = self.W / 2

        self.source_order = []
        self.source_positions = []

        source_list = metadata['sources']
        for source in source_list:
            source_id = source['order']
            source_position = source['position']
            
            self.source_order.append(source_id)
            self.source_positions.append(source_position)

        return self

    def get_metadata(self):
        metadata = {}
        
        metadata['duration'] = self.D
        metadata['sampling_rate'] = self.fs
        metadata['max_order'] = self.max_order
        
        metadata['n_mics'] = self.M
        metadata['absorption'] = self.absorption
        metadata['mic_distance'] = self.mic_distance
        metadata['mic_positions'] = self.mic_positions
        
        room_desc = {}
        room_desc['length'] = self.L
        room_desc['width'] = self.W
        metadata['room'] = room_desc

        metadata['sources'] = []
        for i, source_id in enumerate(self.source_order):
            source = {'position':self.source_positions[i], 'order':source_id}
            metadata['sources'].append(source)
        
        return metadata

    def save(self, path):
        metadata = self.get_metadata()
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def simulate(self,
                 source_audio):
        """
        Input: list of source_audio (T,)
        returns y (M, T)
        """
        
        corners = np.array([[self.left_wall, self.bottom_wall],
                            [self.right_wall, self.bottom_wall],
                            [self.right_wall, self.top_wall],
                            [self.left_wall, self.top_wall]]).T

        room = pra.room.Room.from_corners(corners,
                              absorption=self.absorption,
                              fs=self.fs,
                              max_order=self.max_order)

        mic_array = np.array(self.mic_positions).T#pra.circular_2D_array(center=[0., 0.], M=self.M, phi0=180, radius=self.mic_distance * 0.5 * 1e-2)
        room.add_microphone_array(mic_array)

        for i, source_pos in enumerate(self.source_positions):
            room.add_source(source_pos, signal=source_audio[i])

        y = room.simulate(return_premix=True)

        total_samples = int(round(self.D * self.fs))
        return y[..., :total_samples]

    def _get_random_source_pos(self, L, R, B, T, K):
        pos = [0, 0]

        while np.linalg.norm(pos) < K:
            x = np.random.uniform(L, R)
            y = np.random.uniform(B, T)

            pos = [x, y]

        return pos


def test():
    n_sources = 5
    duration = 1
    save_path = 'mymetadata.json'
    
    simulator = PRASimulator().initialize_room_with_random_params(n_sources, duration)
    simulator.save(save_path)

    simulator2 = PRASimulator().intialize_from_metadata(save_path)
    
    x = [np.random.random(44100) for i in range(n_sources)]
    # x = [np.sin(2 * np.pi * 440 * np.arange(0, 1, 1/44100)) for i in range(n_sources)]
    y = simulator2.simulate(x) * 1e3

    import soundfile as sf
    sf.write('audio.wav', y[0].T, 44100)


if __name__ == "__main__":
    test()
