import React from 'react';
import XRayScene from '../three/XRayScene';

export default function XRaySpline({ splineUrl, ...props }) {
  if (splineUrl) {
    return (
      <iframe
        src={splineUrl}
        frameBorder="0"
        width="100%"
        height="100%"
        title="X-Ray Spline Scene"
        className="w-full h-full pointer-events-auto"
      />
    );
  }
  return <XRayScene {...props} />;
}
