from dipy.align import imaffine, transforms


def register_affinely(static, moving):
    affine_reg = imaffine.AffineRegistration(metric=None, level_iters=[5000, 500, 50], sigmas=None,
                    factors=None, method='L-BFGS-B', ss_sigma_factor=None, options=None, verbosity=1)
    transform =  transforms.AffineTransform3D()
    params0 = None
    affine_map = affine_reg.optimize(static, moving, transform, params0)
    return affine_map

def transform(moving, affine_map):
    moving = moving.astype(float)
    transformed_image = affine_map.transform(moving, interp='nearest')
    return transformed_image
