import random
from albumentations.core.transforms_interface import BasicTransform
from torch.nn import functional as F
from albumentations import Compose, random_utils
import torch
import numpy as np
import math
import typing


def crop_or_pad(data, max_len=100, mode="start"):
    diff = max_len - data.shape[0]

    if diff <= 0:  # Crop
        if mode == "start":
            data = data[:max_len]
        else:
            offset = np.abs(diff) // 2
            data = data[offset: offset + max_len]
        return data
    
    coef = 0
    padding = torch.ones((diff, data.shape[1], data.shape[2])) * coef
    data = torch.cat([data, padding])
    return data

          
class Resample(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        sample_rate=(0.8,1.2),
        always_apply=False,
        p=0.5,
    ):
        super(Resample, self).__init__(always_apply, p)
        
        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def apply(self, data, sample_rate=1., **params):
        length = data.shape[0]
        new_size = max(int(length * sample_rate),1)
        new_x = F.interpolate(data.permute(1,2,0),new_size).permute(2,0,1)
        return new_x

    def get_params(self):
        return {"sample_rate": random.uniform(self.rate_lower, self.rate_upper)}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply}

class TemporalCrop(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        length=384,
        always_apply=False,
        p=0.5,
    ):
        super(TemporalCrop, self).__init__(always_apply, p)

        self.length = length

    def apply(self, data, length=384,offset_01=0.5, **params):
        l = data.shape[0]
        max_l = np.clip(l-length,1,length)
        offset = int(offset_01 * max_l)
        data = data[offset:offset+length]
        return data

    def get_params(self):
        return {"offset_01": random.uniform(0, 1)}

    def get_transform_init_args_names(self):
        return ("length", )
    
    @property
    def targets(self):
        return {"image": self.apply}    

    
class TemporalMask(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.2,0.4), 
        mask_value=float('nan'),
        num_masks = (1,2),
        always_apply=False,
        p=0.5,
    ):
        super(TemporalMask, self).__init__(always_apply, p)

        self.size = size
        self.num_masks = num_masks
        self.mask_value = mask_value

    def apply(self, data, mask_sizes=[0.3],mask_offsets_01=[0.2], mask_value=float('nan'), **params):
        l = data.shape[0]
        x_new = data.clone()
        for mask_size, mask_offset_01 in zip(mask_sizes,mask_offsets_01):
            mask_size = int(l * mask_size)
            max_mask = np.clip(l-mask_size,1,l)
            mask_offset = int(mask_offset_01 * max_mask)
            x_new[mask_offset:mask_offset+mask_size] = torch.tensor(mask_value)
        return x_new

    def get_params(self):
        num_masks = np.random.randint(self.num_masks[0], self.num_masks[1])
        mask_size = [random.uniform(self.size[0], self.size[1]) for _ in range(num_masks)]
        mask_offset_01 = [random.uniform(0, 1) for _ in range(num_masks)]
        return {"mask_sizes": mask_size,
                'mask_offsets_01':mask_offset_01,
                'mask_value':self.mask_value,}

    def get_transform_init_args_names(self):
        return ("size","mask_value","num_masks")
    
    @property
    def targets(self):
        return {"image": self.apply}  

class TemporalCut(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.2,0.4), 
        num_masks = (1,2),
        always_apply=False,
        p=0.5,
    ):
        super(TemporalCut, self).__init__(always_apply, p)

        self.size = size
        self.num_masks = num_masks

    def apply(self, data, mask_sizes=[0.3],mask_offsets_01=[0.2], **params):
        l = data.shape[0]
        x_new = data.clone()
        final_mask = torch.ones(data.shape[0]).bool()
        for mask_size, mask_offset_01 in zip(mask_sizes,mask_offsets_01):
            mask_size = int(l * mask_size)
            max_mask = np.clip(l-mask_size,1,l)
            mask_offset = int(mask_offset_01 * max_mask)
            final_mask[mask_offset:mask_offset+mask_size] = 0
            
        if final_mask.sum() == 0:
            final_mask[:6] = 1
        print(final_mask)
        return x_new[final_mask]

    def get_params(self):
        num_masks = np.random.randint(self.num_masks[0], self.num_masks[1])
        mask_size = [random.uniform(self.size[0], self.size[1]) for _ in range(num_masks)]
        mask_offset_01 = [random.uniform(0, 1) for _ in range(num_masks)]
        return {"mask_sizes": mask_size,
                'mask_offsets_01':mask_offset_01,
                }

    def get_transform_init_args_names(self):
        return ("size","num_masks")
    
    @property
    def targets(self):
        return {"image": self.apply}  

class TemporalFill(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.2,0.4), 
        num_masks = (1,2),
        always_apply=False,
        p=0.5,
    ):
        super(TemporalFill, self).__init__(always_apply, p)

        self.size = size
        self.num_masks = num_masks

    def apply(self, data, mask_sizes=[0.3],mask_offsets_01=[0.2], **params):
        
        x_new = data.clone()
        for mask_size, mask_offset_01 in zip(mask_sizes,mask_offsets_01):
            l = x_new.shape[0]
            mask_size = int(l * mask_size)
            max_mask = np.clip(l-mask_size,1,l)
            mask_offset = int(mask_offset_01 * max_mask)
            
            x_fill = torch.zeros((mask_size,data.shape[1],data.shape[2]),dtype=data.dtype)
            x_new = torch.cat([x_new[:mask_offset],x_fill,x_new[mask_offset:]])

        return x_new

    def get_params(self):
        num_masks = np.random.randint(self.num_masks[0], self.num_masks[1])
        mask_size = [random.uniform(self.size[0], self.size[1]) for _ in range(num_masks)]
        mask_offset_01 = [random.uniform(0, 1) for _ in range(num_masks)]
        return {"mask_sizes": mask_size,
                'mask_offsets_01':mask_offset_01,
                }

    def get_transform_init_args_names(self):
        return ("size","num_masks")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
class TemporalMaskV2(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.2,0.4), 
        mask_value=float('nan'),
        num_masks = (1,2),
        always_apply=False,
        p=0.5,
    ):
        super(TemporalMaskV2, self).__init__(always_apply, p)

        self.size = size
        self.num_masks = num_masks
        self.mask_value = mask_value

    def apply(self, data, mask_sizes=[0.3],mask_offsets_01=[0.2], mask_value=float('nan'), **params):
        l = data.shape[0]
        x_new = data.clone()
        for mask_size, mask_offset_01 in zip(mask_sizes,mask_offsets_01):
            mask_size = int(l * mask_size)
            max_mask = np.clip(l-mask_size,1,l)
            mask_offset = int(mask_offset_01 * max_mask)
            x_new[mask_offset:mask_offset+mask_size] = torch.tensor(mask_value)
        return x_new

    def get_params(self):
        num_masks = np.random.randint(self.num_masks[0], self.num_masks[1])
        b0 = self.size[0] / num_masks
        b1 = self.size[1] / num_masks
        mask_size = [random.uniform(b0, b1) for _ in range(num_masks)]
        mask_offset_01 = [random.uniform(0, 1) for _ in range(num_masks)]
        return {"mask_sizes": mask_size,
                'mask_offsets_01':mask_offset_01,
                'mask_value':self.mask_value,}

    def get_transform_init_args_names(self):
        return ("size","mask_value","num_masks")
    
    @property
    def targets(self):
        return {"image": self.apply}  

# def spatial_mask(x, size=(0.5,1.), mask_value=float('nan')):
#     mask_offset_y = np.random.uniform(x[...,1].min().item(),x[...,1].max().item())
#     mask_offset_x = np.random.uniform(x[...,0].min().item(),x[...,0].max().item())
#     mask_size = np.random.uniform(size[0],size[1])
#     mask_x = (mask_offset_x<x[...,0]) & (x[...,0] < mask_offset_x + mask_size)
#     mask_y = (mask_offset_y<x[...,1]) & (x[...,1] < mask_offset_y + mask_size)
#     mask = mask_x & mask_y
#     x_new = x.contiguous()
#     x_new = x * (1-mask[:,:,None].float()) #+ mask_value[:,:,None] * mask_value
#     return x_new
    
    
class SpatialMask(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.5,1.), 
        mask_value=float('nan'),
        mode = 'abolute',
        always_apply=False,
        p=0.5,
    ):
        super(SpatialMask, self).__init__(always_apply, p)

        self.size = size
        self.mask_value = mask_value
        self.mode = mode

    def apply(self, data, mask_size=0.75, offset_x_01=0.2, offset_y_01=0.2,mask_value=float('nan'), **params):
        # mask_size absolute width 
        
        
        
        #fill na makes it easier with min and max
        data0 = data.contiguous()
        data0[torch.isnan(data0)] = 0
        
        x_min, x_max = data0[...,0].min().item(), data0[...,0].max().item() 
        y_min, y_max = data0[...,1].min().item(), data0[...,1].max().item() 
        
        if self.mode == 'relative':
            mask_size_x = mask_size * (x_max - x_min)
            mask_size_y = mask_size * (y_max - y_min)
        else:
            mask_size_x = mask_size 
            mask_size_y = mask_size             

        mask_offset_x = offset_x_01 * (x_max - x_min) + x_min
        mask_offset_y = offset_y_01 * (y_max - y_min) + y_min
        
        mask_x = (mask_offset_x<data0[...,0]) & (data0[...,0] < mask_offset_x + mask_size_x)
        mask_y = (mask_offset_y<data0[...,1]) & (data0[...,1] < mask_offset_y + mask_size_y)
        
        mask = mask_x & mask_y
        x_new = data.contiguous() * (1-mask[:,:,None].float()) + mask[:,:,None] * mask_value
        return data

    def get_params(self):
        params = {"offset_x_01": random.uniform(0, 1)}
        params['offset_y_01'] = random.uniform(0, 1)
        params['mask_size'] = random.uniform(self.size[0], self.size[1])
        params['mask_value'] = self.mask_value
        return params

    def get_transform_init_args_names(self):
        return ("size", "mask_value","mode")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
class SpatialMaskFix(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.5,1.), 
        mask_value=float('nan'),
        mode = 'abolute',
        always_apply=False,
        p=0.5,
    ):
        super(SpatialMaskFix, self).__init__(always_apply, p)

        self.size = size
        self.mask_value = mask_value
        self.mode = mode

    def apply(self, data, mask_size=0.75, offset_x_01=0.2, offset_y_01=0.2,mask_value=float('nan'), **params):
        # mask_size absolute width 
        
        
        
        #fill na makes it easier with min and max
        data0 = data.contiguous()
        data0[torch.isnan(data0)] = 0
        
        x_min, x_max = data0[...,0].min().item(), data0[...,0].max().item() 
        y_min, y_max = data0[...,1].min().item(), data0[...,1].max().item() 
        
        if self.mode == 'relative':
            mask_size_x = mask_size * (x_max - x_min)
            mask_size_y = mask_size * (y_max - y_min)
        else:
            mask_size_x = mask_size 
            mask_size_y = mask_size             

        mask_offset_x = offset_x_01 * (x_max - x_min) + x_min
        mask_offset_y = offset_y_01 * (y_max - y_min) + y_min
        
        mask_x = (mask_offset_x<data0[...,0]) & (data0[...,0] < mask_offset_x + mask_size_x)
        mask_y = (mask_offset_y<data0[...,1]) & (data0[...,1] < mask_offset_y + mask_size_y)
        
        mask = mask_x & mask_y
        x_new = data.contiguous() * (1-mask[:,:,None].float()) + mask[:,:,None] * mask_value
        return x_new

    def get_params(self):
        params = {"offset_x_01": random.uniform(0, 1)}
        params['offset_y_01'] = random.uniform(0, 1)
        params['mask_size'] = random.uniform(self.size[0], self.size[1])
        params['mask_value'] = self.mask_value
        return params

    def get_transform_init_args_names(self):
        return ("size", "mask_value","mode")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
class SpatialMaskV2(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        size=(0.05,0.1), 
        mask_value=0,
        num_masks = (1,2),
        mode = 'relative',
        always_apply=False,
        p=0.5,
    ):
        super(SpatialMaskV2, self).__init__(always_apply, p)

        self.size = size
        self.mask_value = mask_value
        self.num_masks = num_masks
        self.mode = mode

    def apply(self, data, mask_sizes=[0.75], offsets_x_01=[0.2], offsets_y_01=[0.2],mask_value=float('nan'), **params):
        # mask_size absolute width 
        
        
        
        #fill na makes it easier with min and max
        data0 = data.contiguous()
        data0[torch.isnan(data0)] = 0
        
        x_min, x_max = data0[...,0].min().item(), data0[...,0].max().item() 
        y_min, y_max = data0[...,1].min().item(), data0[...,1].max().item() 
        
        x_new = data.contiguous()
        for mask_size,offset_x_01,offset_y_01  in zip(mask_sizes,offsets_x_01,offsets_y_01):
        
            mask_size_x = mask_size * (x_max - x_min)
            mask_size_y = mask_size * (y_max - y_min)


            mask_offset_x = offset_x_01 * (x_max - x_min) + x_min
            mask_offset_y = offset_y_01 * (y_max - y_min) + y_min

            mask_x = (mask_offset_x<data0[...,0]) & (data0[...,0] < mask_offset_x + mask_size_x)
            mask_y = (mask_offset_y<data0[...,1]) & (data0[...,1] < mask_offset_y + mask_size_y)
        
            mask = mask_x & mask_y
            x_new = x_new.contiguous() * (1-mask[:,:,None].float()) + mask[:,:,None] * mask_value
        return x_new

    def get_params(self):
        
        num_masks = np.random.randint(self.num_masks[0], self.num_masks[1])
        mask_sizes = [random.uniform(self.size[0], self.size[1]) for _ in range(num_masks)]
        offsets_x_01 = [random.uniform(0, 1) for _ in range(num_masks)]
        offsets_y_01 = [random.uniform(0, 1) for _ in range(num_masks)]
        
        
        params = {"offsets_x_01": offsets_x_01,
                  "offsets_y_01":offsets_y_01,
                  "mask_sizes":mask_sizes,
                  "mask_value":self.mask_value}
                 

        return params

    def get_transform_init_args_names(self):
        return ("size", "mask_value","mode","num_masks")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
class SpatialNoise(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        noise_range=(-0.05,0.05), 
        always_apply=False,
        p=0.5,
    ):
        super(SpatialNoise, self).__init__(always_apply, p)

        self.noise_range = noise_range

    def apply(self, data, noise, **params):
        # mask_size absolute width 
        
        data = data + torch.tensor(noise, dtype=data.dtype)
        return data
    
    def get_params_dependent_on_targets(self, params):
        data = params["image"]
        noise = random_utils.uniform(self.noise_range[0],self.noise_range[1],data.shape)

        return {"noise": noise}

    def get_transform_init_args_names(self):
        return ("noise_range",)
    
    @property
    def targets_as_params(self):
        return ["image"]
    
    @property
    def targets(self):
        return {"image": self.apply}
    
def spatial_random_affine(data,scale=None,shear=None,shift=None,degree=None,center=(0,0)):
    
    data_tmp = None
    
    #if input is xyz, split off z and re-attach later
    if data.shape[-1] == 3:
        data_tmp = data[...,2:]
        data = data[...,:2]
        
    center = torch.tensor(center)
    
    if scale is not None:
        data = data * scale
        
    if shear is not None:
        shear_x, shear_y = shear
        shear_mat = torch.tensor([[1.,shear_x],
                                  [shear_y,1.]])    
        data = data @ shear_mat
        center = center + torch.tensor([shear_y, shear_x])
        
    if degree is not None:
        data -= center
        radian = degree/180*np.pi
        c = math.cos(radian)
        s = math.sin(radian)
        
        rotate_mat = torch.tensor([[c,s],
                                   [-s, c]])
        
        data = data @ rotate_mat
        data = data + center
        
    if shift is not None:
        data = data + shift
                          
    if data_tmp is not None:
        data = torch.cat([data,data_tmp],axis=-1)
        
    return data    
    
class SpatialAffine(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        scale (float, float) or None
        
        
        
    
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        scale  = None,
        shear = None,
        shift  = None,
        degree = None,
        center_xy = (0,0),
        always_apply=False,
        p=0.5,
    ):
        super(SpatialAffine, self).__init__(always_apply, p)

        self.scale  = scale
        self.shear  = shear
        self.shift  = shift
        self.degree  = degree
        self.center_xy = center_xy

    def apply(self, data, scale=None,shear=None,shift=None,degree=None,center=(0,0), **params):
        
        new_x = spatial_random_affine(data,scale=scale,shear=shear,shift=shift,degree=degree,center=center)
        return new_x

    def get_params(self):
        params = {'scale':None, 'shear':None, 'shift':None, 'degree':None,'center_xy':self.center_xy}
        if self.scale:
            params['scale']= random.uniform(self.scale[0], self.scale[1])
        if self.shear:
            
            shear_x = shear_y = random.uniform(self.shear[0],self.shear[1])
            if random.uniform(0,1) < 0.5:
                shear_x = 0.
            else:
                shear_y = 0.     
            params['shear']= (shear_x, shear_y)
        if self.shift:
            params['shift']= random.uniform(self.shift[0], self.shift[1])
        if self.degree:
            params['degree']= random.uniform(self.degree[0], self.degree[1])
        
        return params

    def get_transform_init_args_names(self):
        return ("scale", "shear", "shift", "degree")
    
    @property
    def targets(self):
        return {"image": self.apply}
    
class SpatialAffineCone(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        scale (float, float) or None
        
        
        
    
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        scale  = None,
        shear = None,
        shift  = None,
        degree = None,
        center_xy = (0,0),
        num_windows=(6,7),
        always_apply=False,
        p=0.5,
    ):
        super(SpatialAffineCone, self).__init__(always_apply, p)

        self.scale  = scale
        self.shear  = shear
        self.shift  = shift
        self.degree  = degree
        self.center_xy = center_xy
        self.num_windows = num_windows

    def apply(self, data, scale=None,shear=None,shift=None,degree=None,center=(0,0),num_windows=6, **params):
        
        scale0, scale1 = scale
        shear0, shear1 = shear
        shift0, shift1 = shift
        degree0, degree1 = degree
        num_windows  = min(num_windows,data.shape[0])#min windows and len(data)
        
        r = torch.arange(num_windows) / num_windows
        scales = (1-r) * scale0 + r * scale1
        shears_x = (1-r) * shear0[0] + r * shear1[0]
        shears_y = (1-r) * shear0[1] + r * shear0[1]
        shifts = (1-r) * shift0 + r * shift1
        degrees = (1-r) * degree0 + r * degree1
        
        datas = torch.tensor_split(data,num_windows,dim=0)
        new_x = []
        for i in range(len(datas)):
            new_x += [spatial_random_affine(datas[i],scale=scales[i],shear=(shears_x[i],shears_y[i]),shift=shifts[i],degree=degrees[i],center=center)]
        new_x = torch.cat(new_x, dim=0)
        return new_x

    def get_params(self):
        params = {'scale':None, 'shear':None, 'shift':None, 'degree':None,'center_xy':self.center_xy}
        if self.scale:
            params['scale']= (random.uniform(self.scale[0], self.scale[1]),random.uniform(self.scale[0], self.scale[1]))
        if self.shear:
            
            shear_x = shear_y = random.uniform(self.shear[0],self.shear[1])
            if random.uniform(0,1) < 0.5:
                shear_x = 0.
            else:
                shear_y = 0.     
                
            shear_x2 = shear_y2 = random.uniform(self.shear[0],self.shear[1])
            if random.uniform(0,1) < 0.5:
                shear_x2 = 0.
            else:
                shear_y2 = 0.     
                
            params['shear']= ((shear_x, shear_y),(shear_x2, shear_y2))
            
            
            
            
        if self.shift:
            params['shift']= (random.uniform(self.shift[0], self.shift[1]),random.uniform(self.shift[0], self.shift[1]))
        if self.degree:
            params['degree']= (random.uniform(self.degree[0], self.degree[1]),random.uniform(self.degree[0], self.degree[1]))
        params['num_windows'] = np.random.randint(self.num_windows[0],self.num_windows[1])
        return params

    def get_transform_init_args_names(self):
        return ("scale", "shear", "shift", "degree","num_windows")
    
    @property
    def targets(self):
        return {"image": self.apply}
    
class FingersDrop(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        n_fingers, 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(FingersDrop, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        hand_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in 'x_left_ x_right_'.split()]
        hand_indices = np.array(hand_indices)
        self.finger_indices = np.reshape(hand_indices[:,1:], (-1, 4))
        if type(n_fingers) == int:
            self.n_fingers = (n_fingers,n_fingers+1)
        else:
            self.n_fingers = n_fingers
        self.mask_value = mask_value

    def apply(self, data,fidx=None, **params):
        x_new = data.contiguous()
        
        # Drop fingers
#         n_fingers = np.random.randint(self.n_fingers[0])
#         fidx = np.random.randint(len(self.finger_indices), size=self.n_fingers)
        
        drop_indices = self.finger_indices[fidx].flatten()
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        n_fingers = np.random.randint(self.n_fingers[0],self.n_fingers[1])
        fidx = np.random.randint(len(self.finger_indices), size=self.n_fingers)
        params = {
                  'fidx':fidx}
        return params

    def get_transform_init_args_names(self):
        return ( "n_fingers","mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
class OnLandmarkIds(BasicTransform):
    """Select one of transforms to apply. Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

    def __init__(self, transform,
                 landmark_indices,
                always_apply=False,
                p=1.0,):
        super(OnLandmarkIds, self).__init__(always_apply, p)
        self.transform = transform
        self.transform_p = transform.p
        self.landmark_indices = landmark_indices

    
    def apply(self, data, force_apply= False, **params):
        x_new = data.clone()     
        
        if self.transform_p and random.random() < self.p:
            t = self.transform
            x_new[:,self.landmark_indices] = t(image=x_new[:,self.landmark_indices],force_apply= True)['image']
        return x_new    
    
    
    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ("landmark_indices",)
    
    @property
    def targets(self):
        return {"image": self.apply}  

    
class PoseDrop(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(PoseDrop, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        self.pose_indices = np.array([t for t,l in enumerate(landmarks) if 'pose' in l])
        self.mask_value = mask_value

    def apply(self, data,pidx=None, **params):
        x_new = data.contiguous()
        
        drop_indices = self.pose_indices[pidx].flatten()
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        pidx = range(len(self.pose_indices)) #all pose idxs
        params = {'pidx':pidx}
        return params

    def get_transform_init_args_names(self):
        return ( "mask_value",)
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
class TimeShift(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        shift_rate=(-10,10),
        always_apply=False,
        p=0.5,
    ):
        super(TimeShift, self).__init__(always_apply, p)
        
        rate_lower = shift_rate[0]
        rate_upper = shift_rate[1]

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def apply(self, data, shift_rate=5, **params):
        length = data.shape[0]
        
        if shift_rate > 0:
            zeros = torch.zeros((shift_rate,data.shape[1],data.shape[2]),dtype=data.dtype)
            new_x = torch.cat([zeros,data.clone()])
        elif shift_rate > -data.shape[0]+2:
            new_x = data.clone()[-shift_rate:]
        else:
            new_x = data.clone()

        return new_x

    def get_params(self):
        return {"shift_rate": np.random.randint(self.rate_lower, self.rate_upper)}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper")
    
    @property
    def targets(self):
        return {"image": self.apply}

    
class FaceDrop(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(FaceDrop, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        self.face_indices = np.array([t for t,l in enumerate(landmarks) if 'face' in l])
        self.mask_value = mask_value

    def apply(self, data,pidx=None, **params):
        x_new = data.contiguous()
        
        drop_indices = self.face_indices[pidx].flatten()
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        pidx = range(len(self.face_indices)) #all pose idxs
        params = {'pidx':pidx}
        return params

    def get_transform_init_args_names(self):
        return ("mask_value",)
    
    @property
    def targets(self):
        return {"image": self.apply}  

class OnWindows(BasicTransform):
    """Select one of transforms to apply. Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

    def __init__(self, transform,
                 window_size = (0.4,0.6),
                 num_windows = (1,2),
                always_apply=False,
                p=1.0,):
        super(OnWindows, self).__init__(always_apply, p)
        self.transform = transform
        self.transform_p = transform.p
        self.num_windows = num_windows
        self.window_size = window_size

    
    def apply(self, data, window_sizes=[0.5],window_offsets_01=[0.25], force_apply= False, **params):
        l = data.shape[0]
        x_new = data.clone()     
        
        if self.transform_p and random.random() < self.p:
            t = self.transform
#             print(window_sizes,window_offsets_01)
            for mask_size, mask_offset_01 in zip(window_sizes,window_offsets_01):
                mask_size = int(l * mask_size)
                max_mask = np.clip(l-mask_size,1,l)
                mask_offset = int(mask_offset_01 * max_mask)            
                x_new[mask_offset:mask_offset+mask_size] = t(image=x_new[mask_offset:mask_offset+mask_size],force_apply= True)['image']
        return x_new    
    
    
    def get_params(self):
        num_windows = np.random.randint(self.num_windows[0],self.num_windows[1])
        window_sizes = [random.uniform(self.window_size[0], self.window_size[1]) for _ in range(num_windows)]
        window_offsets_01 = [random.uniform(0, 1) for _ in range(num_windows)]
        return {"num_windows": num_windows,
               "window_sizes": window_sizes,
                "window_offsets_01":window_offsets_01
               }

    def get_transform_init_args_names(self):
        return ("window_size", "num_windows")
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
    
class OneOf(BasicTransform):
    """Select one of transforms to apply. Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

    def __init__(self, transforms, 
                always_apply=False,
                p=1.0,):
        super(OneOf, self).__init__(always_apply, p)
        self.transforms = transforms
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:

        if self.transforms_ps and random.random() < self.p:
            idx: int = np.random.choice(range(len(self.transforms)), p=self.transforms_ps, size = 1)[0]
            t = self.transforms[idx]
            data = t(force_apply=True, **data)
        return data
    
    def get_params(self):
        return {}

    def get_transform_init_args_names(self):
        return ()
    
    @property
    def targets(self):
        return {"image": self.apply}  

    
class DynamicResample(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        rate (float,float): lower and upper amount of resampling rate. Should both be float

    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        sample_rate=(0.8,1.2),
        windows = (5,10),
        always_apply=False,
        p=0.5,
    ):
        super(DynamicResample, self).__init__(always_apply, p)
        
        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError("Invalid combination of rate_lower and rate_upper. Got: {}".format((rate_lower, rate_upper)))
        
        self.rate_lower = rate_lower
        self.rate_upper = rate_upper
        self.windows = windows

    def apply(self, data, sample_rates=[1.], **params):
        
        sample_rates = sample_rates[:data.shape[0]] #handle very short seq e.g. seq_len=6 
        
        chunks = data.chunk(len(sample_rates))
        new_x = []
        for sample_rate, chunk in zip(sample_rates,chunks):
            length = chunk.shape[0]
            new_size = max(int(length * sample_rate),1)
            new_x += [F.interpolate(chunk.permute(1,2,0),new_size).permute(2,0,1)]
        new_x = torch.cat(new_x)
        return new_x

    def get_params(self):
        w = np.random.randint(self.windows[0],self.windows[1])
        sample_rates = [random.uniform(self.rate_lower, self.rate_upper) for _ in range(w)]
        return {"sample_rates": sample_rates,}

    def get_transform_init_args_names(self):
        return ("rate_lower", "rate_upper","windows")
    
    @property
    def targets(self):
        return {"image": self.apply}
    
    
class PoseDrop2(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(PoseDrop2, self).__init__(always_apply, p)
        landmarks = [i for i in landmarks if 'x_' in i]
        pose_indices = np.array([t for t,l in enumerate(landmarks) if 'pose' in l])
        pose_indices = pose_indices.reshape(-1, 2).T
        self.pose_indices_type1 = pose_indices[:,0:].T.reshape(-1)
        self.pose_indices_type2 = pose_indices[:,1:].T.reshape(-1)
        self.pose_indices_type3 = pose_indices[:,2:].T.reshape(-1)
        self.pose_indices_type4 = pose_indices[:,3:].T.reshape(-1)
                
        self.mask_value = mask_value

    def apply(self, data, **params):
        x_new = data.contiguous()
        
        pose_indices = random.choice([self.pose_indices_type1,
                                      self.pose_indices_type2,
                                      self.pose_indices_type3,
                                      self.pose_indices_type4])
        drop_indices = pose_indices.flatten()
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        #pidx = range(len(self.pose_indices)) #all pose idxs
        #params = {'pidx':pidx}
        params = {}
        return params

    def get_transform_init_args_names(self):
        return ( "mask_value",)
    
    @property
    def targets(self):
        return {"image": self.apply}  
    
class HandDrop2(BasicTransform):
    """
    stretches/ squeezes input over time dimension
    
    Args:
        landmarks : xyz_landmarks .. array of strings
        n_fingers : num finger droppped, finger dropped for whole sequence
    Targets:
        image

    Image types:
        float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)
    """

    def __init__(
        self,
        landmarks, 
        mask_value=float('nan'),
        always_apply=False,
        p=1.0,
    ):
        super(HandDrop2, self).__init__(always_apply, p)
        
        landmarks = [i for i in landmarks if 'x_' in i]
        hand_indices = [[t for t,l in enumerate(landmarks) if i in l] for i in 'x_left_ x_right_'.split()]
        self.hand_indices = np.array(hand_indices).flatten()
        self.mask_value = mask_value

    def apply(self, data, fidx=None, **params):
        x_new = data.contiguous()
        
        drop_indices = self.hand_indices
        x_new[:, drop_indices] =  torch.tensor(self.mask_value)
        
        return x_new

    def get_params(self):
        params = {}
        return params

    def get_transform_init_args_names(self):
        return ( "n_fingers","mask_value")
    
    @property
    def targets(self):
        return {"image": self.apply} 