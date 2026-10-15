import React from 'react';
import HumanAnatomy from '../three/HumanAnatomy';

/**
 * Spline Adapter component for Hero 3D scene.
 * Can be swapped with @splinetool/react-spline when a Spline exported scene URL is provided.
 */
export default function HeroSpline({ splineUrl, ...props }) {
  if (splineUrl) {
    return (
      <div className="w-full h-full relative" aria-label="Spline 3D Medical Scene">
        {/* Placeholder for @splinetool/react-spline integration */}
        <iframe
          src={splineUrl}
          frameBorder="0"
          width="100%"
          height="100%"
          title="Spline 3D Scene"
          className="w-full h-full pointer-events-auto"
        />
      </div>
    );
  }

  // Native React Three Fiber implementation
  return <HumanAnatomy {...props} />;
}
