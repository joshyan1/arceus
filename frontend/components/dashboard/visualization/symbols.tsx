export function InputPlus({ active }: { active: boolean }) {
  const stroke = active ? "#ea580c" : "#A8A29E";

  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <circle cx="8" cy="8" r="7.5" stroke={stroke} />
      <line x1="8" y1="2.5" x2="8" y2="13.5" stroke={stroke} />
      <line x1="13.5" y1="8" x2="2.5" y2="8" stroke={stroke} />
    </svg>
  );
}

export function PosEncoding({ active }: { active: boolean }) {
  const stroke = active ? "#ea580c" : "#A8A29E";

  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <circle cx="12" cy="12" r="11.5" stroke={stroke} />
      <path
        d="M12 12C12 8.82436 9.42564 6.25 6.25 6.25C3.07436 6.25 0.5 8.82436 0.5 12"
        stroke={stroke}
        strokeWidth="0.958333"
      />
      <path
        d="M23.5 12C23.5 15.1756 20.9256 17.75 17.75 17.75C14.5744 17.75 12 15.1756 12 12"
        stroke={stroke}
        strokeWidth="0.958333"
      />
    </svg>
  );
}
