import Image from 'next/image';
import '../app/globals.css';

const ImagePage = () => {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1 style={{ padding: '20px', fontWeight: 'bold', fontSize: '2rem' }}>Activation Unique Stems Histogram</h1>
      <Image
        src="/activation_unique_stems_histogram.png"
        alt="activation by unique stems histogram"
        width={1120}
        height={640}
      />
    </main>
  );
};

export default ImagePage;
