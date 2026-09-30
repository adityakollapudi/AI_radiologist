import React from 'react';
import MRIAnatomy from '../three/MRIAnatomy';

export default function MRISpline({ splineUrl, ...props }) {
  if (splineUrl) {
    return (
      <iframe
        src={splineUrl}
        frameBorder="0"
        width="100%"
        height="100%"
        title="MRI Spline Scene"
        className="w-full h-full pointer-events-auto"
      />
    );
  }
  return <MRIAnatomy {...props} />;
}
