import { useState } from 'react';
import axios from 'axios';
import LoadingOverlay from 'react-loading-overlay';

import healthy from '../assets/images/dancing.png';

const ImageUpload = () => {
  const [file, setFile] = useState(null);
  const [imageError, setimageError] = useState(null);
  const [estimatedImage, setEstimatedImage] = useState(null);
  const [isActive, setIsActive] = useState(false);

  const handleImageChange = (event) => {
    if (event.target.files[0]) {
      var pattern = /image-*/;

      if (!event.target.files[0].type.match(pattern)) {
        setFile(null);
        setimageError('*Invalid image format');
        return;
      }
    }
    setFile(URL.createObjectURL(event.target.files[0]));
    setimageError(null);
    handleImageSubmit(event.target.files[0]);
  };

  const handleImageSubmit = async (image) => {
    try {
      const data = new FormData();
      data.append('file', image);
      setIsActive(true);
      const estimatedResponse = await axios.post('http://192.168.1.102:5000/api/predict', data, {
        responseType: 'blob'
      });

      setEstimatedImage(URL.createObjectURL(estimatedResponse.data));
      setIsActive(false);
    } catch (err) {
      console.log(err);
    }
  };

  return (
    <LoadingOverlay active={isActive} spinner text='Loading your content...'>
      <div className='container py-5'>
        <header className='text-white text-center'>
          <h1 className='display-4'>Human Pose Tracking</h1>
          <p className='lead mb-0'>Whole-body 3D Pose Reconstruction and Estimation.</p>
          <p className='mb-5 font-weight-light'>
            Based on Deep Neural Network
            <a href='https://bootstrapious.com' className='text-white'>
              <u></u>
            </a>
          </p>
          <img src={healthy} alt='' width='150' className='mb-4' />
        </header>
        <div className='row py-4'>
          <div className='col-lg-6 mx-auto'>
            <div className='input-group mb-3 px-2 py-2 rounded-pill bg-white shadow-sm hvr-grow'>
              <input
                id='upload'
                type='file'
                onChange={(event) => handleImageChange(event)}
                className='form-control border-0'
                accept='image/*'
              />
              <label id='upload-label' htmlFor='upload' className='font-weight-light text-muted'>
                Choose file
              </label>
              <div className='input-group-append hvr-grow'>
                <label htmlFor='upload' className='btn btn-light m-0 rounded-pill px-4'>
                  <i className='fa fa-cloud-upload mr-2 text-muted'></i>
                  <small className='text-uppercase font-weight-bold text-muted '>Choose file</small>
                </label>
              </div>
            </div>

            <p className='font-italic text-white text-center'>
              The estimated image will be rendered inside the box below.
            </p>
            <div className='image-area mt-4'>
              <img
                id='imageResult'
                src={estimatedImage || file}
                alt=''
                className='img-fluid rounded shadow-sm mx-auto d-block'
              />
            </div>
          </div>
        </div>
      </div>
    </LoadingOverlay>
  );
};

export default ImageUpload;
