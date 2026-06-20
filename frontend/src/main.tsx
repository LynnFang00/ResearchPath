import React from 'react';
import ReactDOM from 'react-dom/client';
import { SearchPage } from './pages/SearchPage';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <SearchPage />
  </React.StrictMode>,
);
