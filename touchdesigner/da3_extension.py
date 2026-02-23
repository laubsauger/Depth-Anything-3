"""
Depth Anything 3 Extension for TouchDesigner

Main extension class that sets up parameters and provides control methods.

Usage:
1. Create a COMP container
2. Add this script as Text DAT named 'DepthAnything3Ext'
3. Set Extension Object parameter to: op('./DepthAnything3Ext').module.DepthAnything3Ext(me)
4. Extension will set up all parameters and provide control methods

Resolution TOP Setup (for aspect-aware downsampling):
1. Create Resolution TOP named 'resolution_preprocess'
2. Set Resolution -> Resolution to 'Custom'
3. Width Expression: parent().ext.DepthAnything3Ext.GetProcessResWidth()
4. Height Expression: parent().ext.DepthAnything3Ext.GetProcessResHeight()
5. Wire: [video source] -> resolution_preprocess -> in_video
"""


class DepthAnything3Ext:
    """
    Main extension for Depth Anything 3 streaming in TouchDesigner.
    Sets up all parameters and provides control methods.
    """

    def __init__(self, ownerComp):
        """
        Initialize extension.

        Args:
            ownerComp: The COMP this extension is attached to
        """
        self.ownerComp = ownerComp
        self.setupParameters()

    def setupParameters(self):
        """
        Setup all parameters on the parent COMP.
        Called during initialization.
        """
        # Clear existing custom pages
        for page in self.ownerComp.customPages:
            page.destroy()

        # Connection Page (includes backend, control, status, and input)
        page = self.ownerComp.appendCustomPage('Connection')

        # Connection section
        p = page.appendStr('Host', label='Host')
        p[0].default = 'localhost'
        p[0].val = 'localhost'

        p = page.appendInt('Port', label='Port')
        p[0].default = 8080
        p[0].min = 1
        p[0].max = 65535
        p[0].clampMin = True
        p[0].clampMax = True
        p[0].val = 8080

        # Input section
        p = page.appendInt('Frameskip', label='Frame Skip (N)')
        p[0].default = 0
        p[0].min = 0
        p[0].max = 10
        p[0].clampMin = True
        p[0].clampMax = True
        p[0].startSection = True
        p[0].val = 0

        p = page.appendToggle('Usenumpy', label='Send as Numpy')
        p[0].default = True
        p[0].val = True

        p = page.appendInt('Syncbuffersize', label='Sync Buffer Size')
        p[0].default = 30
        p[0].min = 10
        p[0].max = 120
        p[0].clampMin = True
        p[0].clampMax = True
        p[0].val = 30

        # Backend section
        p = page.appendStr('Backendpath', label='Backend Path')
        p[0].default = '/Users/flo/work/code/Depth-Anything-3'
        p[0].startSection = True
        p[0].val = '/Users/flo/work/code/Depth-Anything-3'

        p = page.appendStr('Modeldir', label='Model Directory')
        p[0].default = 'depth-anything/DA3-SMALL'
        p[0].val = 'depth-anything/DA3-SMALL'

        p = page.appendInt('Backendport', label='Backend Port')
        p[0].default = 8080
        p[0].val = 8080

        p = page.appendToggle('Autorestart', label='Auto Restart Backend')
        p[0].default = True
        p[0].val = True

        # Control section
        p = page.appendPulse('Reconnect', label='Reconnect')
        p[0].startSection = True

        page.appendPulse('Restartbackend', label='Restart Backend')
        page.appendPulse('Stopbackend', label='Stop Backend')

        # Status section
        p = page.appendStr('Connectionstatus', label='Connection Status')
        p[0].default = 'Disconnected'
        p[0].readOnly = True
        p[0].startSection = True
        p[0].val = 'Disconnected'

        p = page.appendStr('Backendstatus', label='Backend Status')
        p[0].default = 'Unknown'
        p[0].readOnly = True
        p[0].val = 'Unknown'

        p = page.appendInt('Framessent', label='Frames Sent')
        p[0].default = 0
        p[0].readOnly = True
        p[0].val = 0

        p = page.appendInt('Framesreceived', label='Frames Received')
        p[0].default = 0
        p[0].readOnly = True
        p[0].val = 0

        p = page.appendFloat('Currentfps', label='Current FPS')
        p[0].default = 0.0
        p[0].readOnly = True
        p[0].val = 0.0

        # Model Configuration Page
        page = self.ownerComp.appendCustomPage('Model')

        p = page.appendMenu('Model', label='Model')
        p[0].menuNames = ['DA3-SMALL', 'DA3-BASE', 'DA3-LARGE', 'DA3-GIANT', 'DA3NESTED-GIANT-LARGE']
        p[0].menuLabels = ['DA3-SMALL (80M - Fastest)', 'DA3-BASE (120M - Balanced)',
                            'DA3-LARGE (350M - Quality)', 'DA3-GIANT (1.15B - Slow)',
                            'DA3NESTED-GIANT-LARGE (1.4B - Metric+GS)']
        p[0].default = 0  # DA3-SMALL for real-time
        p[0].val = 0

        p = page.appendMenu('Device', label='Device')
        p[0].menuNames = ['mps', 'cuda', 'cpu']
        p[0].menuLabels = ['MPS (Apple Silicon)', 'CUDA (NVIDIA)', 'CPU (Slow)']
        p[0].default = 0  # MPS
        p[0].val = 0

        # Stream Configuration Page
        page = self.ownerComp.appendCustomPage('Stream')

        p = page.appendInt('Windowsize', label='Window Size')
        p[0].default = 1  # Real-time mode
        p[0].min = 1
        p[0].max = 32
        p[0].clampMin = True
        p[0].clampMax = True
        p[0].val = 1

        p = page.appendInt('Overlap', label='Overlap')
        p[0].default = 0  # No overlap for real-time
        p[0].min = 0
        p[0].max = 16
        p[0].clampMin = True
        p[0].clampMax = True
        p[0].val = 0

        p = page.appendMenu('Processres', label='Process Resolution')
        p[0].menuNames = ['252', '378', '504', '756', '1008']
        p[0].menuLabels = ['252 (Fastest)', '378 (Fast)', '504 (Balanced)', '756 (Quality)', '1008 (Max)']
        p[0].default = 0  # 252 for real-time
        p[0].val = 0

        p = page.appendFloat('Maxfps', label='Max FPS')
        p[0].default = 30.0  # Real-time target
        p[0].min = 1.0
        p[0].max = 60.0
        p[0].clampMin = True
        p[0].clampMax = True
        p[0].val = 30.0

        p = page.appendInt('Quality', label='Quality (100=raw float32)')
        p[0].default = 100
        p[0].min = 50
        p[0].max = 100
        p[0].clampMin = True
        p[0].clampMax = True
        p[0].val = 100

        # Visualization
        page = self.ownerComp.appendCustomPage('Visualization')

        p = page.appendToggle('Normalize', label='Auto Normalize')
        p[0].default = True
        p[0].val = True

        p = page.appendToggle('Colorize', label='Colorize Depth')
        p[0].default = True
        p[0].val = True

        p = page.appendMenu('Colormap', label='Color Map')
        p[0].menuNames = ['viridis', 'plasma', 'inferno', 'magma', 'turbo', 'gray']
        p[0].menuLabels = ['Viridis (Blue-Green-Yellow)', 'Plasma (Purple-Pink-Yellow)',
                            'Inferno (Black-Red-Yellow)', 'Magma (Black-Purple-White)',
                            'Turbo (Rainbow)', 'Grayscale']
        p[0].default = 0  # viridis
        p[0].val = 0

        p = page.appendToggle('Invert', label='Invert Depth')
        p[0].default = False
        p[0].val = False

        p = page.appendFloat('Brightness', label='Brightness')
        p[0].default = 1.0
        p[0].min = 0.0
        p[0].max = 3.0
        p[0].clampMin = True
        p[0].clampMax = True
        p[0].val = 1.0

        p = page.appendFloat('Contrast', label='Contrast')
        p[0].default = 1.0
        p[0].min = 0.0
        p[0].max = 3.0
        p[0].clampMin = True
        p[0].clampMax = True
        p[0].val = 1.0

    def onInitTD(self):
        """
        Called at end of frame that this extension is initialized.
        Automatically configures WebSocket to use parameter expressions.
        """
        # Configure WebSocket DAT to use expressions for dynamic parameter updates
        ws = self.ownerComp.op('websocket1')
        if ws:
            # Set netaddress and port parameters to expressions that reference parent parameters
            ws.par.netaddress.expr = 'f"ws://{parent().par.Host}/stream?window_size={parent().par.Windowsize}&overlap={parent().par.Overlap}&max_fps={parent().par.Maxfps}&quality={parent().par.Quality}"'
            ws.par.netaddress.mode = ParMode.EXPRESSION

            ws.par.port.expr = 'parent().par.Port'
            ws.par.port.mode = ParMode.EXPRESSION

            print("Configured WebSocket netaddress and port expressions")
        else:
            print("WARNING: websocket1 DAT not found - create it and reconnect")

    def onPulse(self, par):
        """
        Called when a pulse parameter is triggered.

        Args:
            par: The parameter that was pulsed
        """
        if par.name == 'Reconnect':
            self.Reconnect()
        elif par.name == 'Restartbackend':
            self.RestartBackend()
        elif par.name == 'Stopbackend':
            self.StopBackend()

    def Reconnect(self):
        """
        Reconnect WebSocket with current parameters.
        Called when Reconnect pulse parameter is triggered.
        """
        control_dat = self.ownerComp.op('da3_stream_control')
        if control_dat:
            # Import and call reconnect function from control script
            control_dat.module.reconnect(self.ownerComp)
        else:
            print("ERROR: da3_stream_control DAT not found")

    def RestartBackend(self):
        """
        Restart the backend server.
        Called when Restartbackend pulse parameter is triggered.
        """
        control_dat = self.ownerComp.op('da3_stream_control')
        if control_dat:
            # Import and call restart_backend function from control script
            control_dat.module.restart_backend(self.ownerComp)
        else:
            print("ERROR: da3_stream_control DAT not found")

    def StopBackend(self):
        """
        Stop the backend server.
        Called when Stopbackend pulse parameter is triggered.
        """
        control_dat = self.ownerComp.op('da3_stream_control')
        if control_dat:
            # Import and call stop_backend function from control script
            control_dat.module.stop_backend(self.ownerComp)
        else:
            print("ERROR: da3_stream_control DAT not found")

    def GetProcessResWidth(self):
        """
        Get process resolution width respecting input aspect ratio.
        Call from expression: parent().ext.DepthAnything3Ext.GetProcessResWidth()
        """
        # Get menu name (string like '252', '378', etc.) and convert to int
        process_res_str = self.ownerComp.par.Processres.eval()
        process_res = int(process_res_str)

        # Get input aspect ratio from in_video
        in_video = self.ownerComp.op('in_video')
        if not in_video:
            return process_res

        aspect = in_video.width / in_video.height if in_video.height > 0 else 1.0

        if aspect > 1.0:
            # Landscape: width is limiting dimension
            return process_res
        else:
            # Portrait: scale width by aspect
            return int(process_res * aspect)

    def GetProcessResHeight(self):
        """
        Get process resolution height respecting input aspect ratio.
        Call from expression: parent().ext.DepthAnything3Ext.GetProcessResHeight()
        """
        # Get menu name (string like '252', '378', etc.) and convert to int
        process_res_str = self.ownerComp.par.Processres.eval()
        process_res = int(process_res_str)

        # Get input aspect ratio from in_video
        in_video = self.ownerComp.op('in_video')
        if not in_video:
            return process_res

        aspect = in_video.width / in_video.height if in_video.height > 0 else 1.0

        if aspect > 1.0:
            # Landscape: scale height by aspect
            return int(process_res / aspect)
        else:
            # Portrait: height is limiting dimension
            return process_res
