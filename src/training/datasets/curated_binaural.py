from src.training.datasets.semaudio_binaural_base import SemAudioBinauralBaseDataset

class CuratedBinauralDataset(SemAudioBinauralBaseDataset):
    """
    Torch dataset object for synthetically rendered spatial data.
    """
    labels = [
            "alarm_clock", "baby_cry", "birds_chirping", "cat", "car_horn", 
            "cock_a_doodle_doo", "cricket", "computer_typing", 
            "dog", "glass_breaking", "gunshot", "hammer", "music", 
            "ocean", "door_knock", "singing", "siren", "speech", 
            "thunderstorm", "toilet_flush"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = [
            "alarm_clock", "baby_cry", "birds_chirping", "cat", "car_horn", 
            "cock_a_doodle_doo", "cricket", "computer_typing", 
            "dog", "glass_breaking", "gunshot", "hammer", "music", 
            "ocean", "door_knock", "singing", "siren", "speech", 
            "thunderstorm", "toilet_flush"]
