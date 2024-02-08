class Telem():
    def __init__(self) -> None:
        
        self.lat = None
        self.lon = None
        self.alt = None
        
        self.roll = None
        self.pitch = None
        self.yaw = None
        
        self.roll_rate = None
        self.pitch_rate = None
        self.yaw_rate = None
        
        self.x = None
        self.y = None
        self.z = None
        
        self.vx = None
        self.vy = None
        self.vz = None
        
        self.heading = None
        self.timestamp = None
        
    def __str__(self):
        return f'lat: {self.lat}, lon: {self.lon}, alt: {self.alt}, roll: {self.roll}, pitch: {self.pitch}, yaw: {self.yaw}, roll_rate: {self.roll_rate}, pitch_rate: {self.pitch_rate}, yaw_rate: {self.yaw_rate}, x: {self.x}, y: {self.y}, z: {self.z}, vx: {self.vx}, vy: {self.vy}, vz: {self.vz}, heading: {self.heading}, timestamp: {self.timestamp}'