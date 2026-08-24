import type { Preview } from '@storybook/react-vite'
import { ConfigProvider, theme } from 'antd'
import '../src/styles.css'

const preview: Preview = {
  decorators: [
    (Story) => <ConfigProvider theme={{ algorithm: theme.darkAlgorithm }}><Story /></ConfigProvider>,
  ],
  parameters: {
    a11y: { test: 'error' },
    backgrounds: { default: 'console', values: [{ name: 'console', value: '#09131f' }] },
    layout: 'padded',
  },
}

export default preview
