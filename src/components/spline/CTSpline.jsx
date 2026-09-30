import React from 'react';
import CTScene from '../three/CTScene';

export default function CTSpline({ splineUrl, ...props }) {
  if (splineUrl) {
    return (
      <iframe
        src={splineUrl}
        frameBorder="0"
        width="100%"
        height="100%"
        title="CT Spline Scene"
        className="w-full h-full pointer-events-auto"
      />
    );
  }
  return <CTScene {...props} />;
}
