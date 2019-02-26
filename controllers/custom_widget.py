from pyqtgraph.opengl import GLViewWidget
import numpy as np


class Display(GLViewWidget):
    def __init__(self, *args, **kwds):
        GLViewWidget.__init__(self, *args, **kwds)


    def __init_sim__(self):
        """
        Create Image item for texture display
        Create customMarkers in SIM mode
        """
        self.z_stack = 0
        self.z_offset = 0
        self.metaData = {"SizeX":0,"SizeY":0,"SizeZ":0}
        self.sim = image(data=None, smooth=False, filename=r"\image")
        self.sim_raycast = raycast(filename=r"\image")
        self.sim.setVisible(False)
        self.sim_raycast.setVisible(False)
        self.addItem(self.sim)

    def set_sim_image(self, image):
        if image != None:
            if self._twoDim:
                self.sim.set_data(image.data_rgba_2d,
                                  flip_ud=image.flip["UpsideDown"], flip_lr=image.flip["LeftRight"])
                self.sim.update()
                self.sim.setVisible(True)
            else:
                dat = [color[self.opts['slices'][0]:self.opts['slices'][1]] for color in dat]
                self.sim3D.set_data(np.array(dat), image.color, chNumb,
                                  flip_ud=image.flip["UpsideDown"], flip_lr=image.flip["LeftRight"])
                self.sim3D.update()
                self.sim3D.setVisible(True)
                y = self.sim3D.viewTransform()
                z = y.column(2)
                z.setZ(pxSize['SizeZ']*1000*self.sim3D.data.shape[1]/2)
                y.setColumn(2,z)
                self.sim3D.setTransform(y)
                if not self.sim3D.scaled:
                    self.sim3D.scale(
                        pxSize['SizeX']*self.sim3D.data.shape[2]/2*1000,
                        pxSize['SizeY']*self.sim3D.data.shape[3]/2*1000, 1,)
                    self.sim3D.translate(
                        pxSize['SizeX']*1000*self.sim3D.data.shape[2]/2,
                        pxSize['SizeY']*1000*self.sim3D.data.shape[3]/2,
                        0)
                    self.sim3D.scaled = True
        else:
            self.sim.setVisible(False)

    def set_sim_interpolation(self, min, mag):
        self.sim.mag = mag
        self.sim.min = min

    def set_sim_dimension(self, dim):
        if dim == "3D":
            self._twoDim = False
            self.sim.setVisible(False)
            self.main_window.viewer.display.ShowAllChonfocalChannels()
        elif dim == "2D":
            self.sim3D.setVisible(False)
            self._twoDim = True
        self.main_window.viewer.display.ShowAllChonfocalChannels()