import pandas as pd

class DataCollection:
    def __init__(self, filename:str):
        self.data_dict = {
            'x': [],
            'y': [],
            'z': [],
            'V_e': [],
            'V_n': [],
            'V_d': [],
            'phi': [],
            'theta': [],
            'psi': [],
            'p': [],
            'q': [],
            'r': [],
            'vx': [],
            'vy': [],
            'vz': [],
            'timestamp': []
        }
        
        self.filename = filename
        
    def add_data(self, telem) -> None:
        self.data_dict['x'].append(telem.x)
        self.data_dict['y'].append(telem.y)
        self.data_dict['z'].append(telem.z)
        self.data_dict['V_e'].append(telem.vx)
        self.data_dict['V_n'].append(telem.vy)
        self.data_dict['V_d'].append(telem.vz)
        self.data_dict['phi'].append(telem.roll)
        self.data_dict['theta'].append(telem.pitch)
        self.data_dict['psi'].append(telem.yaw)
        self.data_dict['p'].append(telem.roll_rate)
        self.data_dict['q'].append(telem.pitch_rate)
        self.data_dict['r'].append(telem.yaw_rate)
        self.data_dict['vx'].append(telem.vx)
        self.data_dict['vy'].append(telem.vy)
        self.data_dict['vz'].append(telem.vz)
        self.data_dict['timestamp'].append(telem.timestamp)
 

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.data_dict)
    
    def to_csv(self) -> None:
        self.to_df().to_csv(self.filename, index=False)
        print(f"Data saved to {self.filename}")
        